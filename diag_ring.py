"""环上暗点的角向分布诊断：区分「圆环描边」与「一条粗直线蹭到环」。

_ring_outline_frac 只算占比、剔除线穿环的 ±14° 角区。若暗点来自圆环描边，
4 个象限弧应全暗；若来自旁边一条粗直线（=最外圈格子的边框线），暗点只
集中在直线的那一侧。本脚本把环上采样点的明暗按角度打印出来。
"""
import math
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/yangyang/tsume-app")
from recognition import detect as D  # noqa: E402
import fullboard as F  # noqa: E402

N = 19


def ring_profile(gray, cx, cy, r, n=72, thr=120):
    """返回 [(角度, 是否暗, 是否是被剔除的角区)]。"""
    out = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        deg = math.degrees(a) % 90
        skip = deg < 14 or deg > 76
        x, y = int(cx + r * math.cos(a)), int(cy + r * math.sin(a))
        v = gray[y, x] if (0 <= x < gray.shape[1] and 0 <= y < gray.shape[0]) else 255
        out.append((math.degrees(a), v < thr, skip))
    return out


def quad_frac(prof):
    """四个象限（各自只算 14°~76° 有效区）的暗点比例。"""
    q = [[], [], [], []]
    for deg, dark, skip in prof:
        if skip:
            continue
        q[int(deg // 90) % 4].append(dark)
    return [float(np.mean(v)) if v else 0.0 for v in q]


def bar(prof):
    """把环画成角度条带：# 暗且计入, + 暗但被剔除, . 亮。"""
    s = ""
    for deg, dark, skip in prof:
        if int(deg) % 10 != 0:
            continue
        s += ("+" if skip else "#") if dark else "."
    return s


def run(name, picks):
    bgr = cv2.imread("photos/%s.png" % name)
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gcls = gray.copy()
    gcls[D._blue_mask(bgr) > 0] = 255
    ggrid = gray.copy()
    ggrid[D._blue_mask_strong(bgr) > 0] = 255
    xs, ys = F.detect_full(ggrid, N)
    s = min(float(np.median(np.diff(xs))), float(np.median(np.diff(ys))))
    r = max(int(s * 0.44), 6)
    print("=" * 74)
    print("%s  格距=%.1f 环半径=%d   (角度条带从 0° 起每 10° 一格, 共 36 格)" % (name, s, r))
    for (i, j, tag) in picks:
        cx, cy = int(round(xs[i])), int(round(ys[j]))
        prof = ring_profile(gcls, cx, cy, r)
        q = quad_frac(prof)
        ring = D._ring_outline_frac(gcls, cx, cy, r)
        print("  %-14s i=%2d j=%2d ring=%.3f 象限暗占比=[%.2f %.2f %.2f %.2f] 高象限数=%d"
              % (tag, i, j, ring, q[0], q[1], q[2], q[3],
                 sum(1 for v in q if v > 0.25)))
        print("       0°-90°  |%s|" % bar([p for p in prof if p[0] < 90]))
        print("      90°-180° |%s|" % bar([p for p in prof if 90 <= p[0] < 180]))
        print("     180°-270° |%s|" % bar([p for p in prof if 180 <= p[0] < 270]))
        print("     270°-360° |%s|" % bar([p for p in prof if p[0] >= 270]))


if __name__ == "__main__":
    # (i, j, 标签)：内部疑似真白 / 边缘疑似假白 / 空格
    run("page1_b0", [
        (14, 3, "内部白?"), (3, 14, "内部白?"), (14, 16, "内部白?"),
        (0, 4, "左边缘假白"), (18, 1, "右边缘假白"), (18, 2, "右边缘假白"),
        (9, 9, "空格"), (5, 5, "空格"), (2, 12, "空格"),
    ])
    run("page1_b2", [
        (2, 3, "内部白?"), (5, 2, "内部白?"),
        (18, 6, "右边缘假白"), (18, 9, "右边缘假白"),
        (9, 9, "空格"),
    ])
