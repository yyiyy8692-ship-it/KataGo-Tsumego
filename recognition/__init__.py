"""围棋题图识别模块（recognition）——对外只暴露这一层，内部实现可随时换。

一次拍照 → 完整结果
--------------------
    from recognition import recognize_problem, recognize_page

    r = recognize_problem("photos/p1.jpg")
    r["black"] / r["white"]   # [(x, y)] 网格坐标，照片原方向
    r["corner"]               # "BR" 墙角（哪两条边是真棋盘边）
    r["confidence"]           # 角判定置信度，<0.25 时 need_confirm=True
    r["board_svg"]            # 电子棋盘 SVG 字符串（可直接塞进网页/讲题卡）

    page = recognize_page("photos/page.jpg")   # 一图多题：先切分再逐题识别
    len(page)                                  # 题数；单题时长度为 1

管线（每层都有文档说明为什么这么写，改之前先读）
----------------------------------------------
1 detect.prep_grid    去斜 → 光照归一化 → 检网格（检圆心+投票双路）→ 延伸 → 吸附
2 detect.recognize    四角精化 → 逆透视拉正 → 全交叉点扫描分类（黑/白/空）
3 corner.locate       判墙角：线宽比（主）+ 出头（平局裁决）+ 同册先验
4 detect.split_boards 一图多题切分：网格线峰值 → 投票估格距 → 间距断块
5 board.board_svg     渲染成简约电子棋盘

已知边界（别当成 bug 反复查）
----------------------------
- 只支持**印刷题图**（有子必盖线 / 线穿白子两种印刷风格）。真实木盘照片不适用。
- 白子靠「描边环 + 亮核心」判定，对 JPEG 压缩极敏感：同一张图另存 jpg 后
  白子可漏 7/7（q100 也漏 5/7），存 png 不漏。**切分裁剪一律写 png**。
- 角判定置信度低于 0.25（corner.CONFIRM_THRESHOLD）必须走人工确认，
  四题实测：p1 0.35 / p2 0.18（低）/ q3 0.43 / q4 手写污染需人工指定。
- q4 有 1 颗幻影白子 + 1 颗漏判（铅笔手写与白子特征重叠），机器分不开，
  由上层「画回确认」UI 兜底。
- 一图多题：切分本身可靠（2/3/4 题版面实测全部切对、顺序正确、单题不误切）；
  但整页拍会让每题变小，像素不足时白子变脆——建议每题格距 ≥ ~100px。
"""
from . import board, corner, detect, render  # noqa: F401
from .board import board_svg  # noqa: F401
from .corner import (CORNERS, CONFIRM_THRESHOLD,  # noqa: F401
                     locate as locate_corner, measure, record_confirmation)
from .detect import (prep_grid, recognize, split_boards,  # noqa: F401
                     classify_by_sweep)
from .render import board_svg as board_svg_classic  # noqa: F401

__all__ = [
    "recognize_problem", "recognize_page",
    "recognize", "split_boards", "prep_grid", "classify_by_sweep",
    "locate_corner", "record_confirmation", "measure",
    "CORNERS", "CONFIRM_THRESHOLD",
    "board_svg", "board_svg_classic",
    "board", "corner", "detect", "render",
]


def recognize_problem(photo_path, force_corner=None, read_digits=False,
                      title=None):
    """单题：识别 + 判墙角 + 出电子棋盘。返回 dict（见模块文档）。

    force_corner：人工指定 "TL"/"TR"/"BL"/"BR"，跳过分类器（低置信兜底用）。
    read_digits：手写编号识别，2026-09-08 起默认关闭（用户判定无价值）。
    """
    rec = detect.recognize(photo_path, read_digits=read_digits)
    loc = corner.locate(photo_path, force=force_corner)
    name = title or photo_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    svg = board.board_svg(name, rec["black"], rec["white"], loc["corner"],
                          cols=rec["cols"], rows=rec["rows"])
    out = dict(rec)
    out.update({
        "photo": photo_path,
        "corner": loc["corner"],
        "confidence": loc["confidence"],
        "need_confirm": loc["need_confirm"],
        "forced_corner": bool(loc.get("forced")),
        "corner_scores": loc.get("scores"),
        "board_svg": svg,
    })
    return out


def recognize_page(photo_path, force_corner=None, read_digits=False):
    """一图多题：先 split_boards 切分，再逐题 recognize_problem。

    单题照片返回长度 1 的列表（split_boards 原样返回原图路径）。
    """
    parts = detect.split_boards(photo_path)
    out = []
    for i, p in enumerate(parts):
        r = recognize_problem(p, force_corner=force_corner,
                              read_digits=read_digits,
                              title="第 %d 题" % (i + 1) if len(parts) > 1 else None)
        r["index"] = i
        out.append(r)
    return out
