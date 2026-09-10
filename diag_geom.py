"""整盘识别数值诊断：把「网格几何偏移」与「分类判据失效」分开定位。

不依赖看图，全部输出为数字与字符矩阵。

诊断项：
A. 切块尺寸 / 格距 / 线间距序列（看有无突变）
B. 交点局部重测误差（分象限统计）——几何是否偏
C. 全局网格 vs 局部重测网格 的分类矩阵对比
D. ring / core / dark_frac 三类的分布可分性——判据是否崩
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/yangyang/tsume-app")
from recognition import detect as D  # noqa: E402
import fullboard as F  # noqa: E402

N = 19


def load(name):
    bgr = cv2.imread("photos/%s.png" % name)
    if bgr is None:
        raise SystemExit("读不到 photos/%s.png" % name)
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gcls = gray.copy()
    gcls[D._blue_mask(bgr) > 0] = 255
    ggrid = gray.copy()
    ggrid[D._blue_mask_strong(bgr) > 0] = 255
    return bgr, gray, gcls, ggrid


def dark_and_lines(ggrid, s):
    """复制 detect_full 的暗掩码构造，并返回横/竖线增强图（用于局部重测）。"""
    bg = cv2.medianBlur(ggrid, 91)
    norm = cv2.divide(ggrid, bg, scale=255)
    dark = ((ggrid < float(np.median(ggrid)) * 0.85) | (norm < 195)).astype(np.uint8) * 255
    k = max(int(round(2.0 * s)), 15)
    vline = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    hline = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    return dark, vline, hline


def local_measure(line_img, orient, pos, a0, a1, s):
    """在 [a0,a1) 区间内，沿垂直方向对 pos 做局部峰位重测。"""
    if orient == "v":
        a0, a1 = max(int(a0), 0), min(int(a1), line_img.shape[0])
        if a1 - a0 < 8:
            return None
        prof = line_img[a0:a1, :].sum(axis=0).astype(np.float64)
    else:
        a0, a1 = max(int(a0), 0), min(int(a1), line_img.shape[1])
        if a1 - a0 < 8:
            return None
        prof = line_img[:, a0:a1].sum(axis=1).astype(np.float64)
    w = max(int(round(0.35 * s)), 3)
    p = int(round(pos))
    lo, hi = max(p - w, 0), min(p + w + 1, len(prof))
    if hi - lo < 3:
        return None
    seg = prof[lo:hi]
    if seg.max() <= 0:
        return None
    k2 = lo + int(np.argmax(seg))
    c0, c1 = max(k2 - 2, lo), min(k2 + 2, hi - 1)
    wgt = prof[c0:c1 + 1] + 1e-6
    return float((np.arange(c0, c1 + 1) * wgt).sum() / wgt.sum())


def curve_pts(vline, hline, xs, ys, s, nseg=5):
    """每条线分 nseg 段局部重测，交点用分段中点的线性插值。"""
    sx = float(np.median(np.diff(xs)))
    sy = float(np.median(np.diff(ys)))
    bv = np.linspace(0, vline.shape[0], nseg + 1)
    bh = np.linspace(0, hline.shape[1], nseg + 1)
    vpos = []
    for x in xs:
        vpos.append([local_measure(vline, "v", x, bv[k], bv[k + 1] + 8, sx) for k in range(nseg)])
    hpos = []
    for y in ys:
        hpos.append([local_measure(hline, "h", y, bh[k], bh[k + 1] + 8, sy) for k in range(nseg)])
    mids_v = [(bv[k] + bv[k + 1]) / 2 for k in range(nseg)]
    mids_h = [(bh[k] + bh[k + 1]) / 2 for k in range(nseg)]
    pts = [[None] * len(xs) for _ in range(len(ys))]
    for i in range(len(xs)):
        for j in range(len(ys)):
            vp = [x if m is None else m for m in vpos[i]]
            hp = [y if m is None else m for m in hpos[j]]
            pts[j][i] = (float(np.interp(ys[j], mids_v, vp)),
                         float(np.interp(xs[i], mids_h, hp)))
    return pts


def matrix_of(black, white, nx=N, ny=N):
    rows = []
    for j in range(ny):
        rows.append("".join("B" if (i, j) in black else "W" if (i, j) in white else "."
                            for i in range(nx)))
    return rows


def quadrant(err, nx=N, ny=N):
    q = {}
    for j in range(ny):
        for i in range(nx):
            key = ("上" if j < ny // 2 else "下") + ("左" if i < nx // 2 else "右")
            q.setdefault(key, []).append(err[j][i])
    return {k: (float(np.median(v)), float(np.mean(v)), float(np.max(v)))
            for k, v in q.items()}


def feat_table(gcls, xs, ys, pts, med):
    """统计 B/W/空 三类在 dark_frac / ring / core 上的分布（中位数与四分位）。"""
    from collections import defaultdict
    buckets = defaultdict(lambda: {"dark": [], "ring": [], "core": []})
    s = min(float(np.median(np.diff(xs))), float(np.median(np.diff(ys))))
    rd = max(int(s * 0.38), 5)
    r_ring = max(int(s * 0.44), 6)
    ch = max(2, int(round(s * 0.064)))
    black, white, _ = F.classify_full(gcls, xs, ys, pts)
    for i in range(len(xs)):
        for j in range(len(ys)):
            fx, fy = pts[j][i]
            cx, cy = int(round(fx)), int(round(fy))
            disc = gcls[max(cy - rd, 0):cy + rd + 1, max(cx - rd, 0):cx + rd + 1]
            dk = float(np.mean(disc < 110)) if disc.size else 0.0
            ring = D._ring_outline_frac(gcls, cx, cy, r_ring)
            core = float(np.median(gcls[max(cy - ch, 0):cy + ch + 1,
                                         max(cx - ch, 0):cx + ch + 1]))
            lab = "B" if (i, j) in black else "W" if (i, j) in white else "."
            buckets[lab]["dark"].append(dk)
            buckets[lab]["ring"].append(ring)
            buckets[lab]["core"].append(core)
    out = {}
    for lab, d in buckets.items():
        out[lab] = {k: (float(np.median(v)),
                        float(np.percentile(v, 25)),
                        float(np.percentile(v, 75))) for k, v in d.items()}
    return out, black, white


def main():
    names = sys.argv[1:] or ["page1_b0", "page1_b1", "page1_b2"]
    for name in names:
        bgr, gray, gcls, ggrid = load(name)
        print("=" * 70)
        print("%s  切块 %dx%d" % (name, bgr.shape[1], bgr.shape[0]))
        s_guess = F.acf_period(
            ((ggrid < float(np.median(ggrid)) * 0.85).astype(np.float64)).sum(axis=1))
        res = F.detect_full(ggrid, N)
        if res is None:
            print("  网格检测失败")
            continue
        xs, ys = res
        sx = float(np.median(np.diff(xs)))
        sy = float(np.median(np.diff(ys)))
        s = min(sx, sy)
        print("  格距 x=%.2f y=%.2f  x间距 min/max=%.1f/%.1f  y 间距 min/max=%.1f/%.1f"
              % (sx, sy, min(np.diff(xs)), max(np.diff(xs)),
                 min(np.diff(ys)), max(np.diff(ys))))
        print("  竖线间距: " + " ".join("%.1f" % d for d in np.diff(xs)))
        print("  横线间距: " + " ".join("%.1f" % d for d in np.diff(ys)))

        dark, vline, hline = dark_and_lines(ggrid, s)
        pts = curve_pts(vline, hline, xs, ys, s)
        err = [[float(np.hypot(pts[j][i][0] - xs[i], pts[j][i][1] - ys[j]))
                for i in range(N)] for j in range(N)]
        flat = [e for row in err for e in row]
        print("  【几何】交点重测偏差 中位=%.2fpx (%.2f 格)  最大=%.2fpx (%.2f 格)"
              % (np.median(flat), np.median(flat) / s, max(flat), max(flat) / s))
        for k, v in sorted(quadrant(err).items()):
            print("      %s象限 中位=%.2fpx (%.2f格) 均值=%.2f 最大=%.2f"
                  % (k, v[0], v[0] / s, v[1], v[2]))

        med = float(np.median(gcls))
        b1, w1, _ = F.classify_full(gcls, xs, ys)
        b2, w2, _ = F.classify_full(gcls, xs, ys, pts)
        print("  【判据】全局网格: 黑%d 白%d   局部重测网格: 黑%d 白%d"
              % (len(b1), len(w1), len(b2), len(w2)))
        fb, fw, _ = F.classify_full(gcls, xs, ys, pts)
        tab, _, _ = feat_table(gcls, xs, ys, pts, med)
        for lab in ("B", "W", "."):
            if lab in tab:
                t = tab[lab]
                print("      %s: dark %.2f[%.2f,%.2f]  ring %.2f[%.2f,%.2f]  core %.0f[%.0f,%.0f]"
                      % (lab, t["dark"][0], t["dark"][1], t["dark"][2],
                         t["ring"][0], t["ring"][1], t["ring"][2],
                         t["core"][0], t["core"][1], t["core"][2]))
        print("  【矩阵】全局网格 (行=从上到下, 列=从左到右)")
        for r in matrix_of(b1, w1):
            print("      " + r)
        print("  【矩阵】局部重测网格")
        for r in matrix_of(b2, w2):
            print("      " + r)


if __name__ == "__main__":
    main()
