"""把棋盘局部打成 ASCII 灰度图——在无法看图时用来直接"目视"核对。

用法: diag_ascii.py <题名> <i0> <i1> <j0> <j1>   （格子索引，闭区间）
字符: ' '纸白  '.'亮  ':'微灰  '+'灰  '*'暗  '#'很暗  '@'墨黑
白子=亮心+细暗环；黑子=大片@；网格线=细线状 * / #
输出在光照归一图上做（消除装订阴影的干扰）。
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/yangyang/tsume-app")
from recognition import detect as D  # noqa: E402
import fullboard as F  # noqa: E402

N = 19
RAMP = [(230, " "), (195, "."), (165, ":"), (130, "+"), (95, "*"), (55, "#"), (0, "@")]


def ch(v):
    for thr, c in RAMP:
        if v >= thr:
            return c
    return "@"


def ascii_view(gcls, gnorm, pts, i0, i1, j0, j1, width=110):
    pad = 10
    xs = [pts[j][i][0] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    ys = [pts[j][i][1] for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]
    x0, x1 = int(min(xs)) - pad, int(max(xs)) + pad
    y0, y1 = int(min(ys)) - pad, int(max(ys)) + pad
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, gnorm.shape[1]), min(y1, gnorm.shape[0])
    crop = gnorm[y0:y1, x0:x1]
    sc = max(1, int(np.ceil(crop.shape[1] / width)))
    small = crop[::sc, ::sc]
    print("  区域 i=%d..%d j=%d..%d  像素 x[%d:%d] y[%d:%d] 缩放1/%d"
          % (i0, i1, j0, j1, x0, x1, y0, y1, sc))
    # 列标尺：每格中心对应的字符列
    marks = {}
    for i in range(i0, i1 + 1):
        marks[int(round((pts[j0][i][0] - x0) / sc))] = str(i % 10)
    ruler = "".join(marks.get(c, " ") for c in range(small.shape[1]))
    print("      " + ruler)
    for r in range(small.shape[0]):
        yy = y0 + r * sc
        tag = ""
        for j in range(j0, j1 + 1):
            if abs(yy - pts[j][i0][1]) < sc:
                tag = "%-3d" % j
                break
        print("  %s |%s|" % (tag, "".join(ch(v) for v in small[r])))


def main():
    name = sys.argv[1]
    i0, i1, j0, j1 = (int(v) for v in sys.argv[2:6])
    import os
    for cand in ("photos/%s.png" % name, "review_paper/%s.png_norm.png" % name,
                 "review_out/%s.png" % name, name):
        if os.path.exists(cand):
            bgr = cv2.imread(cand)
            break
    else:
        raise SystemExit("找不到题图: %s" % name)
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gcls = gray.copy()
    gcls[D._blue_mask(bgr) > 0] = 255
    ggrid = gray.copy()
    ggrid[D._blue_mask_strong(bgr) > 0] = 255
    xs, ys, vline, hline = F.detect_full(ggrid, N, with_lines=True)
    pts = F._curve_grid(vline, hline, xs, ys)
    s0 = min(float(np.median(np.diff(xs))), float(np.median(np.diff(ys))))
    pts, has = F._snap_to_cross(vline, hline, pts, s0)
    pts = F._interp_missing(pts, has, len(xs), len(ys))
    gnorm = F._illum_normalize(gcls)
    bl, wh, _ = F.classify_full(gcls, xs, ys, pts, gray_norm=gnorm)
    print("%s 判黑%d 判白%d" % (name, len(bl), len(wh)))
    print("  本区判定: " + ", ".join(
        "%s(i%d,j%d)" % ("B" if (i, j) in bl else "W" if (i, j) in wh else "·", i, j)
        for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)
        if (i, j) in bl or (i, j) in wh))
    ascii_view(gcls, gnorm, pts, i0, i1, j0, j1)


if __name__ == "__main__":
    main()
