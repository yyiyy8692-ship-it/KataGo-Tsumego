"""答案模式服务层：上传 → 切题识别 → **人工确认** → 点一题算一题。

与「上传即全算」的旧版区别（2026-09-12 用户要求）：
    识别完先停住，把每题的棋形画成棋盘图给人核对；确认无误后由用户**手动点
    某一题**才启动 KataGo。
两个理由：
    1. 一页 6 题全算要 4~6 分钟，而用户往往只要其中几题；
    2. 识别错了还硬算 5 分钟是最浪费的一种失败——先让人看一眼更划算。

对外接口（Flask 路由层只做转发）：
    create_task(file, opts) -> tid      # 落盘 + 后台切分识别（不计算）
    task_view(tid, have)    -> dict     # 任务状态；have=前端已拿到的题号
    enqueue(tid, idx)       -> bool     # 请求计算第 idx 题（进串行队列）

计算为什么必须串行：KataGo 单实例 + 神经网络吃满 CPU/GPU，并发只会互相拖慢。
所以这里用单工作线程 + FIFO 队列，而不是每点一题就开一个线程。
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
_QUEUE = []            # [(tid, idx)] —— 待计算的题，FIFO
_QCV = threading.Condition()

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif",
             ".tif", ".tiff")

# 题的几个状态：detected（已识别待确认）→ queued → running → done / error


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
def create_task(file_storage, opts=None):
    """接收上传文件，落盘并起「识别」线程（**不会**自动开算）。"""
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
         "total": 0, "solved": 0, "pending": 0, "boards": [], "error": None,
         "opts": opts, "kind": None, "notes": []}
    TASKS[tid] = t
    threading.Thread(target=_run_detect, args=(tid,), daemon=True).start()
    return tid


def task_view(tid, have=()):
    """给前端的任务视图。

    have: 前端**已经拿到答案**的题号集合。这些题不再回传 solutions（一张变化图
    上百 KB，每次轮询重发纯属浪费）；棋形确认图同样略过。刷新页面时前端状态
    清空，自然变成全量拉取。
    """
    t = TASKS.get(tid)
    if not t:
        return None
    have = set(have)
    boards = []
    for b in t["boards"]:
        d = {k: v for k, v in b.items() if not k.startswith("_")}
        if b["idx"] in have:
            d.pop("solutions", None)
            d.pop("check_svg", None)
        boards.append(d)
    return {"tid": tid, "src_name": t["src_name"], "created": t["created"],
            "status": t["status"], "stage": t["stage"], "kind": t["kind"],
            "total": t["total"], "solved": t["solved"],
            "pending": t["pending"], "error": t["error"],
            "notes": t.get("notes", []), "boards": boards}


def _run_detect(tid):
    t = TASKS[tid]
    try:
        _detect(t)
        t["status"] = "ready"
        t["stage"] = "识别完成"
    except Exception as e:                      # 任何异常都要落到任务里
        t["status"] = "error"
        t["error"] = "%s: %s" % (type(e).__name__, e)
        t["stage"] = "出错"
        traceback.print_exc()


def _detect(t):
    """切分 + 识别。**不碰 KataGo**——这一步只回答「有几题、棋形长什么样」。"""
    from recognition.detect import split_boards
    from recognize_auto import recognize_auto as recognize
    from recognition.board import board_svg as diagram

    _progress(t, "正在解析文件")
    pages, kind, n_pages = prepare_pages(
        os.path.join(t["dir"], "src_" + os.path.basename(t["src_name"])),
        t["src_name"], t["dir"])
    t["kind"] = kind
    t["pages"] = n_pages

    _progress(t, "正在切分题图")
    jobs = []                                   # [(page_no, crop_path)]
    for p, pno in pages:
        try:
            crops = split_boards(p)
        except Exception as e:
            crops = [p]                         # 切分失败当作整页一题处理
            t["notes"].append(
                "第%d页切分失败（%s），按整页一题识别" % (pno, e))
        for cp in crops:
            jobs.append((pno, cp))
    t["total"] = len(jobs)
    if not jobs:
        raise ValueError("没有检测到任何题图——请正对题目重拍，或换 PDF")

    for i, (pno, cp) in enumerate(jobs):
        _progress(t, "正在识别第 %d/%d 题" % (i + 1, len(jobs)))
        b = {"idx": i, "page": pno, "status": "detected",
             "crop_url": "/file/%s/%s" % (t["tid"], os.path.basename(cp))}
        try:
            cols, rows, black, white, corner = recognize(cp)
        except Exception as e:
            b.update(status="error", error="识别失败：%s" % e)
            t["boards"].append(b)
            continue
        if len(black) + len(white) == 0:
            b.update(status="error",
                     error="没识别到任何棋子——请对准题目重拍，或换清晰的 PDF")
            t["boards"].append(b)
            continue

        is_global = cols >= 15 or corner == "FULL"
        label = "第%d题" % (i + 1) + ("（全局实战题）" if is_global else "")
        b.update(cols=cols, rows=rows, corner=corner,
                 n_black=len(black), n_white=len(white),
                 is_global=is_global, label=label,
                 check_svg=diagram(
                     label, black, white, corner, cols, rows,
                     sub="%d×%d ｜ 黑 %d · 白 %d" % (cols, rows, len(black),
                                                    len(white))))
        # 棋子坐标留在服务端：等用户点了才用，不必下发
        b["_st"] = ([list(s) for s in black], [list(s) for s in white])
        t["boards"].append(b)

    t["solved"] = sum(1 for x in t["boards"] if x["status"] == "error")


def _progress(t, stage, **kw):
    t["stage"] = stage
    t.update(kw)


# ---------------------------------------------------------------- 计算队列
def enqueue(tid, idx):
    """把第 idx 题放进计算队列。只有 detected 状态的题能被点。"""
    t = TASKS.get(tid)
    if not t:
        return False
    if not (0 <= idx < len(t["boards"])):
        return False
    b = t["boards"][idx]
    with _QCV:
        if b["status"] != "detected" or (tid, idx) in _QUEUE:
            return False
        b["status"] = "queued"
        _QUEUE.append((tid, idx))
        t["pending"] = sum(1 for x in t["boards"]
                           if x["status"] in ("queued", "running"))
        _QCV.notify()
    return True


def _worker():
    """单工作线程：一次只算一题（KataGo 单实例，并发只会互相拖慢）。"""
    while True:
        with _QCV:
            while not _QUEUE:
                _QCV.wait()
            tid, idx = _QUEUE.pop(0)
        try:
            _solve_one(tid, idx)
        except Exception:
            traceback.print_exc()


def _set_pending(t):
    t["pending"] = sum(1 for x in t["boards"]
                       if x["status"] in ("queued", "running"))


def _solve_one(tid, idx):
    import solver as S

    t = TASKS.get(tid)
    if not t:
        return
    b = t["boards"][idx]
    with _QCV:
        b["status"] = "running"
        _set_pending(t)
    t["stage"] = "正在计算第 %d 题" % (idx + 1)
    black, white = b["_st"]
    black = [tuple(s) for s in black]
    white = [tuple(s) for s in white]
    cols, rows, corner = b["cols"], b["rows"], b["corner"]
    is_global = b["is_global"]
    opts = t["opts"]
    depth = int(opts.get("depth", 5))
    visits = int(opts.get("visits", 3000))
    tol = float(opts.get("tol", 1.0))

    try:
        if is_global:
            multi = {"solutions": [S.solve(cols, rows, black, white,
                                           to_play="B", depth=1,
                                           visits=visits, kind="global")],
                     "alerts": []}
        else:
            multi = S.solve_multi(cols, rows, black, white, to_play="B",
                                  depth=depth, visits=visits, tol=tol)
        sols = multi["solutions"]
        items = []
        for k, res in enumerate(sols):
            tag = " · 最大一手" if is_global else (
                " · 正解%d/%d（%s）" % (k + 1, len(sols), res["best"])
                if len(sols) > 1 else "")
            items.append({
                "title": b["label"] + tag,
                "svg": S.variation_single(b["label"] + tag, cols, rows, corner,
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
        _cache(os.path.join(t["dir"], "cache"), idx, dict(
            idx=idx, label=b["label"], cols=cols, rows=rows, corner=corner,
            black=[list(s) for s in black], white=[list(s) for s in white],
            is_global=is_global, items=items))
    except Exception as e:
        b.update(status="error", error="计算失败：%s: %s" % (type(e).__name__, e))
        traceback.print_exc()

    with _QCV:
        _set_pending(t)
        t["solved"] = sum(1 for x in t["boards"]
                          if x["status"] in ("done", "error"))
        t["stage"] = ("全部完成（%d/%d）" % (t["solved"], t["total"])
                      if not t["pending"] else
                      "还剩 %d 题在计算" % t["pending"])


def _cache(d, i, payload):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "b%d.json" % i), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


threading.Thread(target=_worker, daemon=True).start()
