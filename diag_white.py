"""白子误判溯源：逐个输出被判白格子的 (ring, core, hf, vf)，区分边缘与内部。

目的：验证 classify_full 里的 `max(hf,vf)<=0.4`（线不可见即判白）是不是假白主因，
以及 ring 单独能否干净分开「白子 / 空格」。
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/yangyang/tsume-app")
from recognition import detect as D  # noqa: E402
import fullboard as F  # noqa: E402

N = 19


def run(name):
    bgr = cv2.imread("photos/%s.png" % name)
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gcls = gray.copy()
    gcls[D._blue_mask(bgr) > 0] = 255
    ggrid = gray.copy()
    ggrid[D._blue_mask_strong(bgr) > 0] = 255
    xs, ys = F.detect_full(ggrid, N)
    s = min(float(np.median(np.diff(xs))), float(np.median(np.diff(ys))))
    med = float(np.median(gcls))
    r_line = int(s * 0.35)
    r_ring = max(int(s * 0.44), 6)
    ch = max(2, int(round(s * 0.064)))
    print("=" * 68)
    print("%s 格距=%.1f  med=%.0f" % (name, s, med))
    rows = []
    for i in range(N):
        for j in range(N):
            cx, cy = int(round(xs[i])), int(round(ys[j]))
            hf, vf = D._line_visibility(gcls, cx, cy, r_line,
                                        i == 0, i == N - 1, j == 0, j == N - 1)
            ring = D._ring_outline_frac(gcls, cx, cy, r_ring)
            core = float(np.median(gcls[max(cy - ch, 0):cy + ch + 1,
                                         max(cx - ch, 0):cx + ch + 1]))
            disc = gcls[max(cy - r_ring, 0):cy + r_ring + 1,
                        max(cx - r_ring, 0):cx + r_ring + 1]
            dk = float(np.mean(disc < 110)) if disc.size else 0.0
            why_hv = max(hf, vf) <= 0.4
            why_ring = ring > 0.12 and core >= med - 15
            if why_hv or why_ring:
                edge = i in (0, N - 1) or j in (0, N - 1)
                rows.append((i, j, ring, core, max(hf, vf), dk, edge,
                             "hv" if why_hv and not why_ring else
                             "ring" if why_ring and not why_hv else "both"))
    # 分类汇总
    from collections import Counter
    c = Counter(r[7] for r in rows)
    print("  判白来源: hv兜底(线不可见)=%d  ring真环=%d  两者都中=%d"
          % (c["hv"], c["ring"], c["both"]))
    for kind in ("hv", "ring", "both"):
        sub = [r for r in rows if r[7] == kind]
        if not sub:
            continue
        e = [r for r in sub if r[6]]
        print("    %-5s n=%2d  边缘%2d 内部%2d | ring中位=%.3f core中位=%.0f hv中位=%.2f"
              % (kind, len(sub), len(e), len(sub) - len(e),
                 np.median([r[2] for r in sub]),
                 np.median([r[3] for r in sub]),
                 np.median([r[4] for r in sub])))
    print("  逐格明细 (i,j,ring,core,maxhv,dark,边缘,来源):")
    for r in sorted(rows, key=lambda t: (t[6], t[0], t[1])):
        print("    i=%2d j=%2d ring=%.3f core=%3.0f hv=%.2f dark=%.2f %s %s"
              % (r[0], r[1], r[2], r[3], r[4], r[5], "边" if r[6] else "内", r[7]))


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["page1_b0", "page1_b1", "page1_b2"]):
        run(n)
