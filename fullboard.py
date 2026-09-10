"""整盘（19 路）网格检测——形态学线提取 + 等间距峰列筛选。

为什么不用现管线（detect_grid/vote 盲检点阵）：整页照里 19 路盘格距只有
17~22px，微斜线的投票能量分散到多个 bin，直方图上大半线峰是 0（实测
page1_b0 的 rowh 19 条线只有 3~4 个非零峰），任何基于它的评分都会被噪声解
赢过。形态学长核开运算（cv2.MORPH_OPEN + 2*s 横/竖核）后投影，一条微斜线
的全部能量集中在 1~2 个 bin，19 条线全部清晰出峰，文字笔画被长核滤除。

流程：
1. 暗像素二值 → 横/竖长核开运算 → 投影剖面
2. 峰检测（thr=0.25*max）
3. acf 估格距 s0
4. 峰对锚定：枚举峰对 (i,j)，跨度 ≈ 18*s0 的等分网格，逐线在 ±0.35s 内
   找实际峰验证，取验证得分最高者
5. 逐线吸附到亚像素峰位 → 二次拟合兜底（透视渐变实测 16→19px/格）
"""
import cv2
import numpy as np

from recognition import detect as D


def _dark_mask(gray):
    med = float(np.median(gray))
    return (gray < med * 0.85).astype(np.uint8) * 255


def _line_proj(dark, s, axis):
    """长核开运算提取线后投影。axis="h" 提横线（返回行剖面）。"""
    k = max(int(round(2.0 * s)), 15)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (k, 1) if axis == "h" else (1, k))
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    proj = lines.sum(axis=1 if axis == "h" else 0).astype(np.float64) / 255.0
    return proj


def _peaks_f(pj, thr_frac=0.10):
    """峰位（亚像素重心）与峰值。

    阈值不能高：线被白子盖断 + 开运算在缺口两侧各损失半个核长，远端线峰
    可低到 0.14*max（实测 page1_b0 右半竖线）。阈值只负责去掉纯噪声，
    真正的筛选靠峰对锚定的 19 条等间距验证——假峰进不了等分网格。
    """
    thr = pj.max() * thr_frac
    out = []
    for i in range(1, len(pj) - 1):
        if pj[i] >= pj[i - 1] and pj[i] >= pj[i + 1] and pj[i] > thr:
            c0, c1 = max(i - 2, 0), min(i + 2, len(pj) - 1)
            w = pj[c0:c1 + 1] + 1e-6
            pos = float((np.arange(c0, c1 + 1) * w).sum() / w.sum())
            out.append((pos, float(pj[i])))
    return out


def acf_period(proj, lo=6, hi=90):
    h = np.asarray(proj, dtype=np.float64)
    h = h - h.mean()
    if h.std() < 1e-9:
        return None
    n = len(h)
    hi = min(hi, n // 3)
    ac = [float((h[:n - k] * h[k:]).sum() /
                (np.linalg.norm(h[:n - k]) * np.linalg.norm(h[k:]) + 1e-9))
          for k in range(lo, hi + 1)]
    ac = np.array(ac)
    pk = [i for i in range(1, len(ac) - 1) if ac[i] >= ac[i - 1] and ac[i] >= ac[i + 1]]
    if not pk:
        return None
    mx = ac[pk].max()
    cand = [p for p in pk if ac[p] >= 0.7 * mx]
    return float(lo + cand[0])


def _quad_fit(vals):
    idx = np.arange(len(vals), dtype=np.float64)
    A = np.vstack([np.ones_like(idx), idx, idx ** 2]).T
    coef, *_ = np.linalg.lstsq(A, vals, rcond=None)
    return coef


def _regularize(vals, s0, n):
    """等间距修复：错解的典型形态是首线吸到盘外强峰（标题文字行）、其余线
    被吸附拉回正确位置——表现为一个越界间隔（实测 1.46*s0）。用合规间隔的
    中位数从中心线性重建，把这类解拉回正确网格，再与原解同台复评。"""
    vals = np.asarray(vals, dtype=np.float64)
    d = np.diff(vals)
    good = d[(d > 0.7 * s0) & (d < 1.3 * s0)]
    if len(good) < n // 2:
        return list(vals)
    s_m = float(np.median(good))
    c = float(vals[n // 2])
    return [c + (i - n // 2) * s_m for i in range(n)]


def _snap_score(pks, vals, s0):
    """逐线吸附并评分：平均峰值 + 缺线惩罚。"""
    score, out = 0.0, []
    med_pk = float(np.median([p[1] for p in pks])) if pks else 1.0
    for v in vals:
        near = [p for p in pks if abs(p[0] - v) <= 0.35 * s0]
        if near:
            p = max(near, key=lambda t: t[1])
            score += p[1]
            out.append(p[0])
        else:
            score -= 0.6 * med_pk      # 缺线惩罚：网格踩到盘外/文字行的标志
            out.append(v)
    return score / len(vals), out


def _refine_candidate(pks, vals, s0, n, rounds=3):
    """吸附 → 鲁棒线性拟合 → 再吸附，迭代收敛。

    必要性：透视渐变（实测格距 16→19px/格）使峰对等分初值在中部偏差达
    6px，单次吸附吸错邻线，真解评分反而输给"首线蹭文字强峰"的错解
    （实测 page1_b0：错解 270.9 > 真解）。鲁棒拟合剔除离群线后线性重建，
    错解的盘外首线（偏离网格 1+ 格）被淘汰，整个候选收敛到真网格。
    """
    idx = np.arange(n, dtype=np.float64)
    cur = list(vals)
    sc = -1.0
    for _ in range(rounds):
        arr = np.asarray(cur, dtype=np.float64)
        keep = np.ones(n, dtype=bool)
        coef = None
        for _ in range(3):
            coef = np.polyfit(idx[keep], arr[keep], 1)
            resid = np.abs(arr - np.polyval(coef, idx))
            new_keep = resid <= max(0.4 * s0, 1.5)
            if new_keep.sum() < 6 or (new_keep == keep).all():
                break
            keep = new_keep
        if coef is None:
            break
        grid = list(np.polyval(coef, idx))
        sc, cur = _snap_score(pks, grid, s0)
    return sc, cur


def detect_full(gray, n=19):
    """返回 (xs, ys)：n 条竖线、n 条横线（gray 坐标）。失败返回 None。"""
    bg = cv2.medianBlur(gray, 91)
    norm = cv2.divide(gray, bg, scale=255)
    dark = ((gray < float(np.median(gray)) * 0.85) | (norm < 195)).astype(np.uint8) * 255

    s_guess = acf_period(dark.sum(axis=1).astype(np.float64))
    if s_guess is None:
        return None
    axes = []
    for axis in ("v", "h"):
        proj = _line_proj(dark, s_guess, axis)
        pks = _peaks_f(proj)
        if len(pks) < n:
            return None
        s0 = acf_period(proj) or s_guess
        best, best_score = None, -1.0
        span_t = (n - 1) * s0
        for i in range(len(pks)):
            for j in range(i + 1, len(pks)):
                span = pks[j][0] - pks[i][0]
                if abs(span - span_t) > 0.12 * span_t:
                    continue
                cand = list(np.linspace(pks[i][0], pks[j][0], n))
                sc, vals = _refine_candidate(pks, cand, s0, n)
                if sc > best_score:
                    best_score, best = sc, vals
        if best is None:
            return None
        # 端点边框精修：网格首末线是粗边框，但投影峰形常分裂成双峰
        # （实测 18.0/21.0/22.9），吸附取错峰后端点偏出边框 3~4px，
        # 采样点落到边框旁纸面上 → 整条边缘行列误判白子。在 ±0.25s 内
        # 重取最强峰（含亚像素重心）把端点压回边框上。
        lo_w, hi_w = 0, len(proj) - 1
        for e in (0, n - 1):
            w0 = int(max(best[e] - 0.25 * s0, lo_w))
            w1 = int(min(best[e] + 0.25 * s0, hi_w))
            if w1 <= w0:
                continue
            seg = proj[w0:w1 + 1]
            b_local = w0 + int(np.argmax(seg))
            c0, c1 = max(b_local - 2, w0), min(b_local + 2, w1)
            wgt = proj[c0:c1 + 1] + 1e-6
            best[e] = float((np.arange(c0, c1 + 1) * wgt).sum() / wgt.sum())
        # 二次拟合兜底：透视渐变下吸附值仍可能有小抖动
        coef = _quad_fit(best)
        m = coef[0] + coef[1] * np.arange(n) + coef[2] * np.arange(n) ** 2
        out = []
        for v, mv in zip(best, m):
            out.append(float(v if abs(v - mv) <= 0.3 * s0 else mv))
        axes.append(out)
    return axes[0], axes[1]


def _edge_ghost_shift(black, white, n, need=5):
    """边缘假白子检测：网格整体平移一格的错解（锚到装订阴影带而非边框，
    实测阴影投影峰值 98 > 边框 78，剖面特征不可分）会留下独特指纹——
    靠外的那条边缘线落在边框外纸面上，整列被判白子（亮核心+描边环蹭到
    边框线）。返回需要平移的 (sx, sy)，无指纹返回 None。"""
    col_w0 = sum(1 for (i, j) in white if i == 0)
    col_wn = sum(1 for (i, j) in white if i == n - 1)
    row_w0 = sum(1 for (i, j) in white if j == 0)
    row_wn = sum(1 for (i, j) in white if j == n - 1)
    col_b0 = sum(1 for (i, j) in black if i == 0)
    col_bn = sum(1 for (i, j) in black if i == n - 1)
    row_b0 = sum(1 for (i, j) in black if j == 0)
    row_bn = sum(1 for (i, j) in black if j == n - 1)
    sx = sy = 0
    if col_w0 >= need and col_b0 == 0:
        sx = +1          # 第 0 列在边框左侧 → 网格右移
    elif col_wn >= need and col_bn == 0:
        sx = -1
    if row_w0 >= need and row_b0 == 0:
        sy = +1
    elif row_wn >= need and row_bn == 0:
        sy = -1
    return (sx, sy) if (sx or sy) else None


def _edge_has_structure(gcls, black, white, xs, ys, edge, n, s):
    """新指纹边是否贴着真实线结构（区分两种平移后果的最终裁决）：
    - 平移有效（b0）：新外侧列落在真边框上/旁，窗口内有框线的暗像素；
    - 平移过头（b1/b2）：新外侧列落在盘外纸面上，窗口内几乎无暗像素。
    返回该边全部白子点位暗像素数的中位数。"""
    med = float(np.median(gcls))
    vals = []
    w = max(int(round(0.6 * s)), 6)
    for (i, j) in white:
        if edge == "L" and i != 0: continue
        if edge == "R" and i != n - 1: continue
        if edge == "T" and j != 0: continue
        if edge == "B" and j != n - 1: continue
        cx, cy = int(round(xs[i])), int(round(ys[j]))
        win = gcls[max(cy - w, 0):cy + w + 1, max(cx - w, 0):cx + w + 1]
        vals.append(int((win < med * 0.85).sum()))
    return float(np.median(vals)) if vals else 0.0


def recognize_full(gray, gcls, n=19):
    """整盘入口：网格检测 → 分类 → 边缘假白子反馈平移（带验证）。

    平移验证的必要性：错位一格的网格（锚到装订阴影带）平移修正后棋子
    检出变全（实测 b0 22→32）；但边缘框线点被 ring 判据误判白时也会触发
    平移信号（b1/b2），此时平移会把网格移出棋盘、棋子骤减——总数不降
    才接受平移，否则维持原网格。
    曲线网格（_curve_grid）实测弊大于利暂不启用：白子描边环判据对 ±1px
    亚像素偏移极敏感（ring 0.13~0.33 跨在 0.12 门槛上），曲线重测的微小
    抖动会让白子成片漏检（b1 白 39→12）。

    返回 (xs, ys, black, white)。"""
    xs, ys = detect_full(gray, n)
    if xs is None:
        return None
    s_x = float(np.median(np.diff(xs)))
    s_y = float(np.median(np.diff(ys)))
    black, white, _ = classify_full(gcls, xs, ys)
    for _ in range(2):
        shift = _edge_ghost_shift(black, white, n)
        if shift is None:
            break
        sx, sy = shift
        xs_t = [x + sx * s_x for x in xs]
        ys_t = [y + sy * s_y for y in ys]
    for _ in range(2):
        shift = _edge_ghost_shift(black, white, n)
        if shift is None:
            break
        sx, sy = shift
        xs_t = [x + sx * s_x for x in xs]
        ys_t = [y + sy * s_y for y in ys]
        b_t, w_t, _ = classify_full(gcls, xs_t, ys_t)
        shift_t = _edge_ghost_shift(b_t, w_t, n)
        if shift_t is not None:
            if (shift_t[0] == -sx and sx != 0) or (shift_t[1] == -sy and sy != 0):
                break      # 平移后指纹指向反方向：来回振荡，两解都有问题
            # 平移后仍有新指纹：若新指纹边贴着真实线结构（框线被弯曲挤出
            # 采样精度，b0 修正后的右边框），平移是有效的，接受；若新指纹
            # 边周围无结构（网格移出棋盘落到纸面上，b1/b2 误移），拒绝。
            # 指纹边映射：建议 sx=-1（左移）的指纹在右缘 R，以此类推。
            edge = ("R", "L", "B", "T")[(0 if sx < 0 else 1 if sx > 0
                                         else 2 if sy < 0 else 3)] if (sx or sy) else "R"
            edge2 = ("R", "L", "B", "T")[(0 if shift_t[0] < 0 else 1 if shift_t[0] > 0
                                          else 2 if shift_t[1] < 0 else 3)]
            struct = _edge_has_structure(gcls, b_t, w_t, xs_t, ys_t, edge2, n,
                                         min(s_x, s_y))
            if struct < 12:
                break
        if len(b_t) + len(w_t) < len(black) + len(white):
            break
        xs, ys = xs_t, ys_t
        black, white = b_t, w_t
    return xs, ys, black, white


def _nearest_peak(pks, v, tol_frac=0.35):
    """v 附近 (±tol*格距由调用方保证) 最近最强峰位；无峰返回 None。"""
    near = [p for p in pks if abs(p[0] - v) <= 8.0]
    if not near:
        return None
    return max(near, key=lambda t: t[1])[0]


def _curve_grid(vline, hline, xs, ys, nseg=5):
    """分段曲线网格：书页装订弯曲使网格线在图像里是曲线（实测边缘竖线
    各高度 x 摆动 9~10px ≈ 0.5 格距，中部基本直），单一 x/y 坐标只在局部
    贴合，弯曲区采样点偏出框线 → 边缘行列误判白子。

    每条竖线按 nseg 段在初值 ±0.3s 窗内重测局部 x，横线同理；交点
    (i, j) 的 x 从竖线分段位置沿 y 插值、y 从横线分段位置沿 x 插值。
    局部测量必须用**线增强图**（长核开运算）：原始暗图在棋子密集区
    峰会落到棋子边缘（实测 b1 白子 39→7），线增强图里棋子已被滤除。
    返回 pts[j][i] = (x, y)。"""
    H = hline.shape[0]
    sx = float(np.median(np.diff(xs)))
    sy = float(np.median(np.diff(ys)))
    ny, nx = len(ys), len(xs)

    def measure(line_img, orient, pos, a0, a1):
        """沿垂直方向在 pos±0.3s 内测局部峰位（在对应方向的线增强图上）。"""
        if orient == "v":
            a0, a1 = max(int(a0), 0), min(int(a1), line_img.shape[0])
            if a1 - a0 < 8:
                return None
            colsum = line_img[a0:a1, :].sum(axis=0).astype(np.float64)
        else:
            a0, a1 = max(int(a0), 0), min(int(a1), line_img.shape[1])
            if a1 - a0 < 8:
                return None
            colsum = line_img[:, a0:a1].sum(axis=1).astype(np.float64)
        s = sx if orient == "v" else sy
        w = max(int(round(0.3 * s)), 3)
        p = int(round(pos))
        lo, hi = max(p - w, 0), min(p + w + 1, len(colsum))
        if hi - lo < 3:
            return None
        seg = colsum[lo:hi]
        if seg.max() <= 0:
            return None
        k = lo + int(np.argmax(seg))
        c0, c1 = max(k - 2, lo), min(k + 2, hi - 1)
        wgt = colsum[c0:c1 + 1] + 1e-6
        return float((np.arange(c0, c1 + 1) * wgt).sum() / wgt.sum())

    seg_bounds_v = np.linspace(0, vline.shape[0], nseg + 1)   # 竖线分段沿 y
    seg_bounds_h = np.linspace(0, hline.shape[1], nseg + 1)   # 横线分段沿 x
    vpos = []
    for x in xs:
        segs = [measure(vline, "v", x, seg_bounds_v[k], seg_bounds_v[k + 1] + 8)
                for k in range(nseg)]
        vpos.append([m if m is not None else x for m in segs])
    hpos = []
    for y in ys:
        segs = [measure(hline, "h", y, seg_bounds_h[k], seg_bounds_h[k + 1] + 8)
                for k in range(nseg)]
        hpos.append([m if m is not None else y for m in segs])

    def interp(pos_arr, a, bounds):
        mids = [(bounds[k] + bounds[k + 1]) / 2 for k in range(nseg)]
        return float(np.interp(a, mids, pos_arr))

    pts = [[None] * nx for _ in range(ny)]
    for i in range(nx):
        for j in range(ny):
            x_ij = interp(vpos[i], ys[j], seg_bounds_v)
            y_ij = interp(hpos[j], xs[i], seg_bounds_h)
            pts[j][i] = (x_ij, y_ij)
    return pts


def probe_full(photo_path):
    """整盘路由探测：acf 格距 <35px 视为 19 路全局题。

    为什么不用 prep_grid 的行列数路由：整页照的整盘切块上 prep_grid 自己
    就会检错（实测 19x19 检成 11x10），不可信；格距是物理量，整页拍摄下
    全局题 17~25px、局部死活题 40px+，分得开。
    返回 (is_full, spacing)。
    """
    bgr = cv2.imread(photo_path)
    if bgr is None:
        return False, None
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        sc = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)))
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(gray, 91)
    norm = cv2.divide(gray, bg, scale=255)
    dark = ((gray < float(np.median(gray)) * 0.85) | (norm < 195)).astype(np.uint8)
    s = acf_period(dark.sum(axis=1).astype(np.float64))
    if s is None:
        return False, None
    return bool(s < 35), float(s)


def recognize_full_board(photo_path, n=19):
    """整盘识别入口（与 detect.recognize 返回结构兼容）。

    自带预处理（缩放/去斜/蓝墨双洗，与 prep_grid 同源）。corner 恒 None
    ——整盘四边都是边界，"两墙两开放"的墙角判定不适用。
    """
    bgr = cv2.imread(photo_path)
    if bgr is None:
        raise ValueError("照片读取失败")
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        sc = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * sc), int(h * sc)))
    bgr, _ = D._deskew_image(bgr)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gcls = gray.copy()
    gcls[D._blue_mask(bgr) > 0] = 255
    # 网格检测用强蓝洗（与 prep_grid 同规）：蓝墨在灰度里是暗色，
    # 会污染暗掩码和投影峰（实测不洗则 detect_full 直接失败）
    ggrid = gray.copy()
    ggrid[D._blue_mask_strong(bgr) > 0] = 255
    out = recognize_full(ggrid, gcls, n)
    if out is None:
        raise ValueError("整盘网格检测失败")
    xs, ys, black, white = out
    return {
        "cols": len(xs), "rows": len(ys),
        "black": sorted(black), "white": sorted(white),
        "digits": [],
        "corner": None,          # 整盘四边全边界
        "confidence": None,
        "need_confirm": True,    # 整盘识别仍需人工核对
    }


def classify_full(gray_cls, xs, ys, pts=None):
    """整盘专属分类：classify_by_sweep 的参数为 ~90px 格距近拍调优，
    20px 格距整页照下描边环只有 ~1px、JPEG 模糊后灰度变淡，
    ring 系统性偏低（实测真白子 0.13~0.33，近拍判据 0.25 会漏一半）。
    整盘模式 ring 门槛降到 0.12，其余判据（亮核心、黑子双条件）不变；
    幻影风险靠画回图人工核对兜底。"""
    black, white, marks = set(), set(), []
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    s = min(dx, dy)
    med = float(np.median(gray_cls))
    nx, ny = len(xs), len(ys)
    rd = max(int(s * 0.38), 5)
    r_line = int(s * 0.35)
    r_ring = max(int(s * 0.44), 6)
    ch = max(2, int(round(s * 0.064)))
    for ix in range(nx):
        for iy in range(ny):
            if pts is not None:
                fx, fy = pts[iy][ix]
            else:
                fx, fy = xs[ix], ys[iy]
            cx, cy = int(round(fx)), int(round(fy))
            disc = gray_cls[max(cy - rd, 0):cy + rd + 1,
                            max(cx - rd, 0):cx + rd + 1]
            dark_frac = float(np.mean(disc < 110)) if disc.size else 0.0
            if dark_frac > 0.5 and D._big_dark_frac(
                    gray_cls, cx, cy, max(int(s * 0.44), 6)) > 0.5:
                black.add((ix, iy))
                marks.append((cx, cy, r_line, "B"))
                continue
            hf, vf = D._line_visibility(gray_cls, cx, cy, r_line,
                                        ix == 0, ix == nx - 1,
                                        iy == 0, iy == ny - 1)
            ring = D._ring_outline_frac(gray_cls, cx, cy, r_ring)
            core = float(np.median(gray_cls[max(cy - ch, 0):cy + ch + 1,
                                            max(cx - ch, 0):cx + ch + 1]))
            if max(hf, vf) <= 0.4 or (ring > 0.12 and core >= med - 15):
                white.add((ix, iy))
                marks.append((cx, cy, r_line, "W"))
    return black, white, marks
