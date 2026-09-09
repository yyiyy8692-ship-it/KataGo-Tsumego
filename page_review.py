"""整页题图 → 逐题切分 → 逐题识别 → 每题一张对照图（原图 | 电子棋盘）。

用途：把一页多题的拍照一次性切成 N 道题，每题产出一张 PNG 供人工核对。
对照图左边是原图裁剪（可看到题号、原始墨迹），右边是识别出的电子棋盘，
底部标注尺寸 / 墙角 / 置信度，一眼能看出哪题识别错了。

用法：
    python page_review.py photos/page1.jpg [输出目录]
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from recognition import recognize_page

# ---- 配色（与 recognition/board.py 的 board_svg 保持一致）----
C_BG = (255, 255, 255)
C_GRID = (201, 205, 209)
C_WALL = (44, 78, 114)
C_TEXT = (43, 43, 43)
C_SUB = (44, 78, 114)
C_BLACK = (42, 38, 32)
C_WHITE = (255, 255, 255)
C_WHITE_STROKE = (192, 196, 200)
CORNER_LABEL = {"TL": "左 + 顶", "TR": "右 + 顶",
                "BL": "左 + 底", "BR": "右 + 底"}
WALL_EDGES = {"TL": ("L", "T"), "TR": ("R", "T"),
              "BL": ("L", "B"), "BR": ("R", "B")}

FONT_CANDIDATES = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def _font(size):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_board(black, white, corner, cols=7, rows=9, cell=46, margin=44,
               name="", scale=2):
    """电子棋盘位图（board_svg 的 PIL 复刻，配色/墙线画法一致）。"""
    over = 7
    w = (cols - 1) * cell + 2 * margin
    h = (rows - 1) * cell + 2 * margin + 64
    S = scale
    img = Image.new("RGB", (w * S, h * S), C_BG)
    d = ImageDraw.Draw(img)

    def px(x, y):
        return (margin + x * cell) * S, (margin + y * cell) * S

    lw = max(1, int(2 * S))
    for x in range(cols):
        d.line([px(x, 0), px(x, rows - 1)], fill=C_GRID, width=lw)
    for y in range(rows):
        d.line([px(0, y), px(cols - 1, y)], fill=C_GRID, width=lw)

    wl, wt = WALL_EDGES[corner]
    sw = max(1, int(cell * 0.16 * S))
    xc = cols - 1 if "R" in (wl, wt) else 0
    yc = rows - 1 if "B" in (wl, wt) else 0
    xf = 0 if xc == cols - 1 else cols - 1
    yf = 0 if yc == rows - 1 else rows - 1
    ax, ay = px(xc, yf)
    bx, by = px(xf, yc)
    kx, ky = px(xc, yc)
    ay += (-over if yf == 0 else over) * S
    bx += (-over if xf == 0 else over) * S
    d.line([(ax, ay), (kx, ky), (bx, by)], fill=C_WALL, width=sw, joint="curve")

    r = cell * 0.44 * S
    for x, y in black:
        cx, cy = px(x, y)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=C_BLACK)
    for x, y in white:
        cx, cy = px(x, y)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=C_WHITE,
                  outline=C_WHITE_STROKE, width=max(1, int(1.5 * S)))

    f1, f2 = _font(17 * S), _font(14 * S)
    ty = (h - 40) * S
    if name:
        bb = d.textbbox((0, 0), name, font=f1)
        d.text(((w * S - (bb[2] - bb[0])) / 2, ty), name, font=f1, fill=C_TEXT)
    sub = "墙角 %s ｜ 黑 %d · 白 %d" % (CORNER_LABEL[corner],
                                       len(black), len(white))
    bb = d.textbbox((0, 0), sub, font=f2)
    d.text(((w * S - (bb[2] - bb[0])) / 2, ty + 26 * S), sub, font=f2,
           fill=C_SUB)
    return img


def make_card(crop_path, res, idx, total):
    """一张对照图：标题 + 左原图 / 右电子棋盘。"""
    src = Image.open(crop_path).convert("RGB")
    bw, bh = src.size
    cell = 46
    cols, rows = res.get("cols", 7), res.get("rows", 9)
    board = draw_board(res["black"], res["white"], res["corner"], cols, rows,
                       cell=cell, name="第 %d 题" % (idx + 1))
    # 等比缩放到统一高度
    H = 620
    def fit(im):
        s = H / im.size[1]
        return im.resize((max(1, int(im.size[0] * s)), H), Image.LANCZOS)
    src, board = fit(src), fit(board)

    pad = 28
    head = 96
    W = pad * 3 + src.size[0] + board.size[0]
    Ht = head + H + pad * 2 + 34
    card = Image.new("RGB", (W, Ht), (247, 248, 250))
    d = ImageDraw.Draw(card)
    card.paste(src, (pad, head))
    card.paste(board, (pad * 2 + src.size[0], head))

    f1, f2 = _font(30), _font(19)
    title = "第 %d 题 / 共 %d 题" % (idx + 1, total)
    d.text((pad, 30), title, font=f1, fill=C_TEXT)
    conf = res.get("confidence") or 0.0
    flag = "  ⚠ 需人工确认墙角" if res.get("need_confirm") else ""
    info = ("%d列 x %d行 ｜ 黑 %d · 白 %d ｜ 墙角 %s ｜ 置信度 %.2f%s"
            % (cols, rows, len(res["black"]), len(res["white"]),
               CORNER_LABEL[res["corner"]], conf, flag))
    d.text((pad, 66), info, font=f2, fill=C_SUB)
    note = "左：原图裁剪（%dx%d）　右：识别结果" % (bw, bh)
    d.text((pad, Ht - 30), note, font=f2, fill=(120, 124, 130))
    return card


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "photos/page1.jpg"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "review_out"
    os.makedirs(outdir, exist_ok=True)

    print("切分：", src)
    res = recognize_page(src)
    print("切出 %d 题" % len(res))
    cards = []
    for i, r in enumerate(res):
        p = os.path.join(outdir, "第%d题.png" % (i + 1))
        make_card(r["photo"], r, i, len(res)).save(p)
        cards.append(p)
        print("  第%d题: %s  %dx%d 黑%d 白%d 墙角%s conf=%.2f %s"
              % (i + 1, os.path.basename(r["photo"]), r["cols"], r["rows"],
                 len(r["black"]), len(r["white"]), r["corner"],
                 r.get("confidence") or 0,
                 "NEED_CONFIRM" if r.get("need_confirm") else ""))
    print("输出目录：", outdir)
    return cards


if __name__ == "__main__":
    main()
