"""实验：19 路整盘网格检测——自相关估格距 + 受限追踪 + 二次拟合兜底。

v2 相对 v1 的三处加固（针对 b1 的周期雪崩）：
1. 周期自适应限幅 [0.75, 1.33]*s0：透视漂移一般 ±20%，一旦吸错就雪崩是 v1 的主病。
2. 峰强门槛：窗口内峰太弱就不吸附，用预测位置，避免被文字/白边带跑。
3. 二次拟合兜底：追踪结果先做 pos=a+b*i+c*i^2 稳健拟合（容纳书页弯曲），
   剔除离群线后用模型重排，再小窗吸附。
"""
import sys

import cv2
import numpy as np

from recognition import detect as D


def acf_period(hist, lo=6, hi=90):
    """自相关估格距：取第一个（最小）显著峰，避免取到 2s 的谐波。"""
    h = np.asarray(hist, dtype=np.float64)
    h = h - h.mean()
    if h.std() < 1e-9:
        return None
    n = len(h)
    hi = min(hi, n // 3)
    ac = []
    for k in range(lo, hi + 1):
        a, b = h[:n - k], h[k:]
        ac.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    ac = np.array(ac)
    peaks = [i for i in range(1, len(ac) - 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
    if not peaks:
        return None
    mx = ac[peaks].max()
    cand = [p for p in peaks if ac[p] >= 0.7 * mx]
    return float(lo + cand[0])


def _peak_at(hist, pred, s, lo, hi, min_val):
    """在 pred±0.38s 内找峰；峰强不足则返回 None（不吸附）。"""
    a, b = int(round(max(pred - 0.38 * s, lo))), int(round(min(pred + 0.38 * s, hi)))
    if b <= a:
        return None
    seg = hist[a:b + 1]
    i = int(np.argmax(seg))
    if seg[i] < min_val:
        return None
    best = a + i
    c0, c1 = max(best - 2, a), min(best + 2, b)
    w = hist[c0:c1 + 1].astype(np.float64) + 1e-6
    return float((np.arange(c0, c1 + 1) * w).sum() / w.sum())


def track(hist, n, s0, seed, span_lo=0, span_hi=None):
    """以 seed 为起点双向追踪 n-1 条线，周期限幅自适应。"""
    hi_len = (len(hist) - 1) if span_hi is None else span_hi
    pk = np.asarray(D._peaks(hist, max(3.0, len(hist) * 0.10)))
    if len(pk):
        pkv = np.array([hist[int(round(p))] for p in pk])
        min_val = 0.40 * float(np.median(pkv))
    else:
        min_val = 0.0

    up, s, cur = [], s0, float(seed)
    for _ in range(n - 1):
        pos = _peak_at(hist, cur + s, s, span_lo, hi_len, min_val)
        if pos is None:
            pos = cur + s
        else:
            s = float(np.clip(abs(pos - cur), 0.75 * s0, 1.33 * s0))
        up.append(pos)
        cur = pos
    dn, s, cur = [], s0, float(seed)
    for _ in range(n - 1):
        pos = _peak_at(hist, cur - s, s, span_lo, hi_len, min_val)
        if pos is None:
            pos = cur - s
        else:
            s = float(np.clip(abs(cur - pos), 0.75 * s0, 1.33 * s0))
        dn.append(pos)
        cur = pos
    return sorted(dn[::-1] + [float(seed)] + up)


def _fit_quad(idx, pos):
    """pos = a + b*i + c*i^2 最小二乘；返回系数。"""
    A = np.vstack([np.ones_like(idx), idx, idx ** 2]).T
    coef, *_ = np.linalg.lstsq(A, pos, rcond=None)
    return coef


def refine_quad(vals, s0, n):
    """剔除离群线 → 二次拟合重排 → 输出 n 条等结构线位。"""
    vals = np.asarray(vals, dtype=np.float64)
    idx = np.arange(len(vals), dtype=np.float64)
    coef = _fit_quad(idx, vals)
    model = coef[0] + coef[1] * idx + coef[2] * idx ** 2
    resid = np.abs(vals - model)
    keep = resid <= max(0.35 * s0, np.median(resid) + 0.25 * s0)
    if keep.sum() >= 5:
        coef = _fit_quad(idx[keep], vals[keep])
    # 重排：以保留线的索引重心为原点，向两端按模型外推到 n 条
    gi = float(np.mean(idx[keep])) if keep.any() else len(vals) / 2.0
    i0 = int(round(gi - (n - 1) / 2.0))
    idx_new = np.arange(i0, i0 + n, dtype=np.float64)
    return [float(coef[0] + coef[1] * i + coef[2] * i ** 2) for i in idx_new]


def _phase_search(hist, n, s0):
    """穷举 (相位 a, 格距 s, 弯曲 c) 选 19 条线总墨量最大的网格。

    p_i = a + i*s + c*(i-(n-1)/2)^2。文字行/边框噪声只能贡献一两条强峰，
    正确网格 19 条全中，总分必然胜出——这就是它比"最强峰当种子"稳的原因。
    """
    h = np.asarray(hist, dtype=np.float64)
    L = len(h)
    mid = (n - 1) / 2.0
    idx = np.arange(n, dtype=np.float64)
    best, best_score = None, -1.0
    for s in np.arange(0.88 * s0, 1.121 * s0, 0.05):
        span = (n - 1) * s
        a_max = L - 1 - span
        if a_max < 0:
            continue
        for a in np.arange(0, a_max, 0.5):
            base = a + idx * s
            for c in (-0.010, -0.005, 0.0, 0.005, 0.010):
                pos = base + c * (idx - mid) ** 2
                pi = np.clip(pos, 0, L - 1)
                score = float(np.interp(pi, np.arange(L), h).sum())
                # 边框锚定：整盘首尾两条是粗边框线，投票值远高于内部线。
                score += 2.0 * (h[int(np.clip(round(pos[0]), 0, L - 1))]
                                + h[int(np.clip(round(pos[-1]), 0, L - 1))])
                if score > best_score:
                    best_score, best = score, pos.copy()
    return best


def detect_full(gray, n=19, iters=2):
    colh, rowh = D._vote_hists(gray)
    out = []
    for hist in (colh, rowh):
        s0 = acf_period(hist)
        if s0 is None:
            return None
        vals = _phase_search(hist, n, s0)
        if vals is None:
            return None
        s_est = float(np.median(np.diff(vals)))
        for _ in range(iters):     # 模型初值 → 小窗吸附 → 再拟合
            snapped = []
            for v in vals:
                p = _peak_at(hist, v, s_est, 0, len(hist) - 1, 0.30)
                snapped.append(v if p is None else p)
            vals = np.asarray(refine_quad(snapped, s_est, n))
        out.append(list(vals))
    return out[0], out[1]


def main():
    names = sys.argv[1:] or ["page1_b0", "page1_b1", "page1_b2"]
    for name in names:
        p = "photos/%s.png" % name
        bgr = cv2.imread(p)
        bgr, _ = D._deskew_image(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gg = gray.copy()
        gg[D._blue_mask_strong(bgr) > 0] = 255
        colh, rowh = D._vote_hists(gg)
        try:
            g = D.prep_grid(p)
            old = "%dx%d" % (len(g["xs"]), len(g["ys"]))
        except Exception as e:
            old = "ERR"
        res = detect_full(gg)
        print("%s crop=%dx%d 自相关 x=%.1f y=%.1f | 现管线 %s"
              % (name, bgr.shape[1], bgr.shape[0],
                 acf_period(colh) or -1, acf_period(rowh) or -1, old))
        if res:
            xs, ys = res
            print("    追踪 %dx%d  格距 x≈%.1f y≈%.1f  x间距CV=%.3f y间距CV=%.3f"
                  % (len(xs), len(ys), np.median(np.diff(xs)), np.median(np.diff(ys)),
                     np.std(np.diff(xs)) / np.mean(np.diff(xs)),
                     np.std(np.diff(ys)) / np.mean(np.diff(ys))))
            img = bgr.copy()   # 必须画在 deskew 后的图上，坐标系才一致
            for x in xs:
                cv2.line(img, (int(round(x)), 0), (int(round(x)), img.shape[0] - 1),
                         (0, 0, 255), 1)
            for y in ys:
                cv2.line(img, (0, int(round(y))), (img.shape[1] - 1, int(round(y))),
                         (255, 0, 0), 1)
            cv2.imwrite("dbg_%s_full.png" % name, img)


if __name__ == "__main__":
    main()
