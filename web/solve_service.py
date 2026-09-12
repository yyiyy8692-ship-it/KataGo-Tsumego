"""答案模式服务层：上传 → 切分 → 识别 → 求解 → 渲染。

与 grader 那套「批改模式」的区别：这里**不问题型、不录孩子的下法**，直接
给正解（双解题给全部正解，全局题只给第 1 手），全部按**黑先**计算。

对外只有三个函数，Flask 路由层薄薄一层调它：
    create_task(file, opts) -> tid        # 落盘 + 起后台线程
    task_view(tid, after)   -> dict       # 任务状态（支持增量取题）

为什么要异步 + 轮询：一页 6 题在 KataGo 3000 visits 下要跑 4~6 分钟，
HTTP 请求不可能挂着等。做法是后台串行跑，每算完一题就把 SVG 写进任务
状态，前端 1.5 秒轮询一次，算一题出一题。
"""
import json
import os
import threading
import traceback
import uuid
from datetime import datetime

from PIL import Image, ImageOps

try:                                  # iPhone 相册直传可能是 HEIC
    import pillow_heif
    pillow_heif.register_heif_opener()
except Exception:                     # 装不上不影响主流程（JPEG/PNG/PDF）
    pass

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS = os.path.join(BASE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)

MAX_EDGE = 4000        # 手机原图最长边限制：再大对识别无益，只会拖慢
MAX_PAGES = 20         # PDF 最多处理的页数
PDF_DPI = 300          # 与实测样本 photos/paper_p1_300.png 一致

TASKS = {}
# KataGo 单实例 + 神经网络吃满 CPU/GPU：任务必须串行，并发只会互相拖慢
_RUN_LOCK = threading.Lock()

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif",
             ".tif", ".tiff")


# ---------------------------------------------------------------- 输入归一化
def _normalize_image(src, dst):
    """手机照片 → 端正的 PNG。

    两个必做动作：
    1. exif_transpose：手机竖拍/横拍的照片靠 EXIF 记录方向，不纠正的话
       棋盘是躺着的，后面所有几何假设全错。照片类 bug 大头在这里。
    2. 限制最长边：iPhone 4800px 原图对 20px 格距的识别没有额外收益，
       只是让 _block_is_grid 慢一倍。
    """
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > MAX_EDGE:
        s = MAX_EDGE / float(max(w, h))
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))),
                       Image.LANCZOS)
    im.save(dst)
    return im.size


def _pdf_pages(src, outdir):
    """PDF 逐页渲染成 PNG。返回 [(png_path, page_no), ...]。"""
    import pymupdf
    doc = pymupdf.open(src)
    outs = []
    for i, page in enumerate(doc):
        if i >= MAX_PAGES:
            break
        pix = page.get_pixmap(dpi=PDF_DPI)
        p = os.path.join(outdir, "page_%02d.png" % (i + 1))
        pix.save(p)
        outs.append((p, i + 1))
    n = doc.page_count
    doc.close()
    return outs, n


def prepare_pages(src, name, outdir):
    """按输入类型归一化为若干「页面图」。返回 (pages, source_kind, page_count)。"""
    ext = os.path.splitext(name)[1].lower()
    os.makedirs(outdir, exist_ok=True)
    if ext == ".pdf" or (ext == "" and _is_pdf(src)):
        pages, total = _pdf_pages(src, outdir)
        return pages, "pdf", total
    if ext not in IMAGE_EXT:
        raise ValueError("不支持的文件格式：%s（支持 照片/图片 与 PDF）" % ext)
    size = _normalize_image(src, os.path.join(outdir, "photo.png"))
    return [(os.path.join(outdir, "photo.png"), 1)], "photo", 1


def _is_pdf(path):
    with open(path, "rb") as f:
        return f.read(5) == b"%PDF-"


# ---------------------------------------------------------------- 任务
def _progress(t, stage, **kw):
    t["stage"] = stage
    t.update(kw)


def create_task(file_storage, opts=None):
    """接收上传文件，落盘并返回 tid（后台线程随即开跑）。"""
    opts = opts or {}
    tid = uuid.uuid4().hex[:10]
    d = os.path.join(UPLOADS, tid)
    os.makedirs(d, exist_ok=True)
    name = (file_storage.filename or "upload").replace("/", "_")
    src = os.path.join(d, "src_" + os.path.basename(name))
    file_storage.save(src)

    t = {"tid": tid, "dir": d, "src_name": name,
         "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
         "status": "running", "stage": "正在读取文件",
         "total": 0, "solved": 0, "boards": [], "error": None,
         "opts": opts, "kind": None}
    TASKS[tid] = t
    threading.Thread(target=_run, args=(tid,), daemon=True).start()
    return tid


def task_view(tid, after=0):
    """给前端的任务视图。

    after: 前端已完整收到的题数。只回传 idx >= after 的题，避免每次轮询都
    重发上百 KB 的 SVG。
    """
    t = TASKS.get(tid)
    if not t:
        return None
    return {"tid": tid, "src_name": t["src_name"], "created": t["created"],
            "status": t["status"], "stage": t["stage"], "kind": t["kind"],
            "total": t["total"], "solved": t["solved"], "error": t["error"],
            "boards": [b for b in t["boards"] if b["idx"] >= after]}


def _run(tid):
    """后台管线。**串行**：多个用户同时提交时排队，互不拖慢。"""
    t = TASKS[tid]
    with _RUN_LOCK:
        try:
            _pipeline(t)
            t["status"] = "done"
            t["stage"] = "完成"
        except Exception as e:                      # 任何异常都要落到任务里
            t["status"] = "error"
            t["error"] = "%s: %s" % (type(e).__name__, e)
            t["stage"] = "出错"
            traceback.print_exc()


def _pipeline(t):
    from recognition.detect import split_boards
    from recognize_auto import recognize_auto as recognize
    import solver as S

    opts = t["opts"]
    depth = int(opts.get("depth", 5))
    visits = int(opts.get("visits", 3000))
    tol = float(opts.get("tol", 1.0))

    _progress(t, "正在解析文件")
    pages, kind, n_pages = prepare_pages(
        os.path.join(t["dir"], "src_" + os.path.basename(t["src_name"])),
        t["src_name"], t["dir"])
    t["kind"] = kind
    t["pages"] = n_pages

    # 先切分，好让前端早点知道有几题（"第 2/6 题"比转圈有信息量）
    _progress(t, "正在切分题图")
    jobs = []                                   # [(page_no, crop_path)]
    for p, pno in pages:
        try:
            crops = split_boards(p)
        except Exception as e:
            crops = [p]                         # 切分失败当作整页一题处理
            t.setdefault("split_note", []).append(
                "第%d页切分失败（%s），按整页一题识别" % (pno, e))
        for cp in crops:
            jobs.append((pno, cp))
    t["total"] = len(jobs)
    if not jobs:
        raise ValueError("没有检测到任何题图——请正对题目重拍，或换 PDF")

    for i, (pno, cp) in enumerate(jobs):
        _progress(t, "第 %d/%d 题：识别棋形" % (i + 1, len(jobs)))
        b = {"idx": i, "page": pno, "status": "running",
             "crop_url": "/file/%s/%s" % (t["tid"], os.path.basename(cp))}
        t["boards"].append(b)
        try:
            cols, rows, black, white, corner = recognize(cp)
        except Exception as e:
            b["status"] = "error"
            b["error"] = "识别失败：%s" % e
            t["solved"] += 1
            continue
        b.update(cols=cols, rows=rows, n_black=len(black), n_white=len(white),
                 corner=corner)
        if len(black) + len(white) == 0:
            b["status"] = "error"
            b["error"] = "没识别到任何棋子——请对准题目重拍，或换清晰的 PDF"
            t["solved"] += 1
            continue

        is_global = cols >= 15 or corner == "FULL"
        b["is_global"] = is_global
        label = "第%d题" % (i + 1) + ("（全局实战题）" if is_global else "")
        b["label"] = label

        _progress(t, "第 %d/%d 题：KataGo 计算" % (i + 1, len(jobs)))
        if is_global:
            multi = {"solutions": [S.solve(cols, rows, black, white,
                                           to_play="B", depth=1,
                                           visits=visits, kind="global")],
                     "alerts": []}
        else:
            multi = S.solve_multi(cols, rows, black, white, to_play="B",
                                  depth=depth, visits=visits, tol=tol)

        # 立刻渲染：算完一题前端就能看到一题
        _progress(t, "第 %d/%d 题：出图" % (i + 1, len(jobs)))
        sols = multi["solutions"]
        items = []
        for k, res in enumerate(sols):
            tag = " · 最大一手" if is_global else (
                " · 正解%d/%d（%s）" % (k + 1, len(sols), res["best"])
                if len(sols) > 1 else "")
            items.append({
                "title": label + tag,
                "svg": S.variation_single(label + tag, cols, rows, corner,
                                          black, white, res),
                "best": res["best"], "lead": res.get("lead"),
                "final_lead": res.get("final_lead"),
                "explain": S.explain(res, to_play="B"),
                "alerts": list(res.get("alerts", [])) + list(
                    multi.get("alerts", [])) if k == 0 else
                    list(res.get("alerts", [])),
            })
        b.update(status="done", lead0=multi.get("lead0"),
                 solutions=items, n_solutions=len(sols),
                 cands=[list(c) for c in multi.get("cands", [])])
        t["solved"] += 1

        _cache(os.path.join(t["dir"], "cache"), i, dict(
            idx=i, label=label, cols=cols, rows=rows, corner=corner,
            black=[list(s) for s in black], white=[list(s) for s in white],
            is_global=is_global, items=items))


def _cache(d, i, payload):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "b%d.json" % i), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
