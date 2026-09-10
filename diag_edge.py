"""边缘假白与局部漏白溯源：对比原始 core 与光照归一后的 core，并测端点线宽。

假设 1（边缘多白）：网格端点线偏出粗边框，采样点落在框外/框内的纸面上 →
   core 亮。判据：端点线处的线宽应明显大于内部线（粗边框 3~5px vs 内部 1~2px）。
假设 2（局部漏白）：装订侧阴影让纸面整体变暗，全局阈值把该区白子全砍掉 →
   用 medianBlur 大核估背景做光照归一，看 core 是否被拉回。
"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/Users/yangyang/tsume-app")
from recognition import detect as D  # noqa: E402
import fullboard as F  # noqa: E402

N = 19


def core_maps(gcls):
    raw = cv2.erode(cv2.medianBlur(gcls, 3), np.ones((5, 5), np.uint8))
    bg = cv2.medianBlur(gcls, 91)
    nrm = np.clip(gcls.astype(np.float32) / np.maximum(bg, 1) * 255.0, 0, 255).astype(np.uint8)
    nrm_core = cv2.erode(cv2.medianBlur(nrm, 3), np.ones((5, 5), np.uint8))
    return raw, nrm_core, bg


def width_at(gray, cx, cy, axis, maxr=7):
    """采样点处沿 axis 方向数连续暗像素（线宽，相对阈值 0.75*局部中位）。"""
    med = float(np.median(gray))
    if axis == "x":
        seg = gray[cy, max(cx - maxr, 0):cx + maxr + 1]
    else:
        seg = gray[max(cy - maxr, 0):cy + maxr + 1, cx]
    return int((seg < med * 0.75).sum())


def report(name, spots):
    bgr = cv2.imread("photos/%s.png" % name)
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gcls = gray.copy()
    gcls[D._blue_mask(bgr) > 0] = 255
    ggrid = gray.copy()
    ggrid[D._blue_mask_strong(bgr) > 0] = 255
    xs, ys, vline, hline = F.detect_full(ggrid, N, with_lines=True)
    pts = F._curve_grid(vline, hline, xs, ys)
    s = min(float(np.median(np.diff(xs))), float(np.median(np.diff(ys))))
    med = float(np.median(gcls))
    raw, nrm, bg = core_maps(gcls)
    cores = [float(raw[int(round(pts[j][i][1])), int(round(pts[j][i][0]))])
             for i in range(N) for j in range(N)]
    h, e = np.histogram(np.asarray(cores), bins=np.arange(60, max(min(med, 250), 70), 10))
    mode = float(e[int(np.argmax(h))] + 5)
    thr = mode + 0.70 * (med - mode)
    print("=" * 78)
    print("%s 格距%.1f med=%.0f 阈值=%.0f  (归一图 med=%.0f)"
          % (name, s, med, thr, float(np.median(nrm))))
    for (i, j, tag) in spots:
        fx, fy = pts[j][i]
        cx, cy = int(round(fx)), int(round(fy))
        hf, vf = D._line_visibility(gcls, cx, cy, int(s * 0.35),
                                    i == 0, i == N - 1, j == 0, j == N - 1)
        print("  %-16s i=%2d j=%2d core=%3.0f core_norm=%3.0f 背景=%3.0f "
              "竖线宽=%d 横线宽=%d hv=%.2f/%.2f %s"
              % (tag, i, j, raw[cy, cx], nrm[cy, cx], bg[cy, cx],
                 width_at(gcls, cx, cy, "x"), width_at(gcls, cx, cy, "y"),
                 hf, vf, "→判白" if raw[cy, cx] >= thr else ""))
    return raw, nrm, bg, thr, pts


if __name__ == "__main__":
    # b2：右上角边缘 i=18 列 + 对照 i=17
    report("page1_b2", [
        (18, 3, "右上边缘"), (18, 4, "右上边缘"), (18, 5, "右上边缘"),
        (18, 6, "右上(未判白)"), (17, 3, "对照i=17"), (17, 4, "对照i=17"),
        (16, 3, "对照i=16"), (13, 3, "内部真白?"), (18, 15, "右下边缘"),
    ])
    # b1：右下角 (18,18) + 左下角区域
    report("page1_b1", [
        (18, 18, "右下角多白"), (17, 18, "对照"), (18, 17, "对照"),
        (2, 14, "左下已检出"), (3, 16, "左下已检出"),
        (1, 15, "左下空白"), (1, 16, "左下空白"), (2, 16, "左下空白"),
        (1, 17, "左下空白"), (2, 17, "左下空白"), (3, 17, "左下空白"),
        (0, 14, "左边缘"), (0, 15, "左边缘"), (0, 16, "左边缘"),
    ])
