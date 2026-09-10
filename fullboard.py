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


def _line_images(dark, s):
    """长核开运算得到的横/竖线增强二值图（曲线网格的局部重测要用）。"""
    k = max(int(round(2.0 * s)), 15)
    vline = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    hline = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    return vline, hline


def detect_full(gray, n=19, with_lines=False):
    """返回 (xs, ys)：n 条竖线、n 条横线（gray 坐标）。失败返回 None。

    with_lines=True 时额外返回 (vline, hline)，供 _curve_grid 局部重测。
    """
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
    if with_lines:
        vline, hline = _line_images(dark, s_guess)
        return axes[0], axes[1], vline, hline
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


def recognize_full(gray, gcls, n=19, curve=True):
    """整盘入口：网格检测 → 曲线网格 → 分类 → 边缘假白反馈平移（带验证）。

    curve=True（默认）启用分段曲线网格：书页弯曲使竖线成曲线（实测摆动
    0.3~0.7 格），直线模型的采样点在弯曲区偏出交点，是"右上角不准"和
    "白子全崩"的根因。旧版把曲线网格关掉是因为当时的白子判据依赖 ring，
    对 ±1px 抖动极敏感（b1 白 39→12）；判据改为 core 主导后容忍度足够，
    曲线网格才敢开。

    平移验证的必要性：错位一格的网格（锚到装订阴影带）平移修正后棋子
    检出变全（实测 b0 22→32）；但边缘框线点被误判白时也会触发平移信号
    （b1/b2），此时平移会把网格移出棋盘、棋子骤减——总数不降才接受平移，
    否则维持原网格。平移时 pts 同步平移，保持与 xs/ys 一致。

    返回 (xs, ys, black, white)。"""
    det = detect_full(gray, n, with_lines=curve)
    if det is None:
        return None
    if curve:
        xs, ys, vline, hline = det
        pts = _curve_grid(vline, hline, xs, ys)
    else:
        xs, ys = det
        pts = None
    s_x = float(np.median(np.diff(xs)))
    s_y = float(np.median(np.diff(ys)))
    black, white, _ = classify_full(gcls, xs, ys, pts)
    for _ in range(2):
        shift = _edge_ghost_shift(black, white, n)
        if shift is None:
            break
        sx, sy = shift
        xs_t = [x + sx * s_x for x in xs]
        ys_t = [y + sy * s_y for y in ys]
        pts_t = ([[(x + sx * s_x, y + sy * s_y) for (x, y) in row] for row in pts]
                 if pts is not None else None)
        b_t, w_t, _ = classify_full(gcls, xs_t, ys_t, pts_t)
        shift_t = _edge_ghost_shift(b_t, w_t, n)
        if shift_t is not None:
            if (shift_t[0] == -sx and sx != 0) or (shift_t[1] == -sy and sy != 0):
                break      # 平移后指纹指向反方向：来回振荡，两解都有问题
            # 平移后仍有新指纹：若新指纹边贴着真实线结构，平移有效接受；
            # 若新指纹边周围无结构（网格移出棋盘落到纸面上），拒绝。
            edge2 = ("R", "L", "B", "T")[(0 if shift_t[0] < 0 else 1 if shift_t[0] > 0
                                          else 2 if shift_t[1] < 0 else 3)]
            struct = _edge_has_structure(gcls, b_t, w_t, xs_t, ys_t, edge2, n,
                                         min(s_x, s_y))
            if struct < 12:
                break
        if len(b_t) + len(w_t) < len(black) + len(white):
            break
        xs, ys, pts = xs_t, ys_t, pts_t
        black, white = b_t, w_t
    return xs, ys, black, white


def _seg_measure(line_img, orient, pos, a0, a1, s):
    """段内局部峰位重测：返回 (相对偏移, 峰强)，无有效峰返回 (0.0, 0.0)。

    只测 pos±0.35s 的小窗——书页弯曲量实测 0.3~0.7 格，窗口再大就会吸到
    邻线（旧版吸飞 373px 就是窗口内没有本线、取到了噪声最大值）。
    """
    if orient == "v":
        a0, a1 = max(int(a0), 0), min(int(a1), line_img.shape[0])
        if a1 - a0 < 8:
            return 0.0, 0.0
        prof = line_img[a0:a1, :].sum(axis=0).astype(np.float64)
    else:
        a0, a1 = max(int(a0), 0), min(int(a1), line_img.shape[1])
        if a1 - a0 < 8:
            return 0.0, 0.0
        prof = line_img[:, a0:a1].sum(axis=1).astype(np.float64)
    w = max(int(round(0.35 * s)), 3)
    p = int(round(pos))
    lo, hi = max(p - w, 0), min(p + w + 1, len(prof))
    if hi - lo < 3:
        return 0.0, 0.0
    seg = prof[lo:hi]
    if seg.max() <= 0:
        return 0.0, 0.0
    k = lo + int(np.argmax(seg))
    c0, c1 = max(k - 2, lo), min(k + 2, hi - 1)
    wgt = prof[c0:c1 + 1] + 1e-6
    peak = float((np.arange(c0, c1 + 1) * wgt).sum() / wgt.sum())
    return peak - float(pos), float(seg.max())


def _fill_and_smooth(off, ok, lim):
    """偏移矩阵：沿段插值补全 → 跨相邻线三点中位平滑 → 限幅。

    跨线平滑的依据：书页弯曲是整页的连续形变，相邻网格线的局部偏移量必然
    接近；单点跳变（吸飞）不会在邻线复现，中位滤波能直接把它踢掉。
    """
    m, k = off.shape
    out = np.zeros_like(off)
    idx = np.arange(k)
    for i in range(m):
        if not ok[i].any():
            continue
        out[i] = np.interp(idx, idx[ok[i]], off[i][ok[i]])
    sm = out.copy()
    for i in range(m):
        lo, hi = max(i - 1, 0), min(i + 2, m)
        sm[i] = np.median(out[lo:hi], axis=0)
    return np.clip(sm, -lim, lim)


def _curve_grid(vline, hline, xs, ys, nseg=6, lim_frac=0.5):
    """分段曲线网格（稳健版）：交点 (i,j) 的真实像素坐标 pts[j][i]。

    为什么必须用：书页沿纵向弯曲，竖线在图像里是曲线——实测竖线沿 y 的
    横向摆动 5~14px（0.3~0.7 格），与棋子半径同量级；而横线只受透视影响，
    间距序列平滑单调。单一 x 坐标的直线模型在弯曲区把采样点甩出交点，
    交点坐标错 → 白子（依赖亚像素精度）全崩，右侧摆动最大处即"右上角不准"。

    三道防吸飞锁（旧版实测吸飞率 2%~9%，b1 左侧 6 条线全飞到 250~374px）：
    1 显著性：峰强 < 该方向中位峰强 45% 的测量点判噪声，弃用
    2 限幅：相对全局线的偏移 > 0.5 格判吸飞，弃用
    3 跨线平滑：相邻网格线偏移量做三点中位，单点跳变被邻线拉回
    弃用点由插值补全，整条线无有效点则退化为全局直线（偏移 0）。
    """
    sx = float(np.median(np.diff(xs)))
    sy = float(np.median(np.diff(ys)))
    nx, ny = len(xs), len(ys)
    bv = np.linspace(0, vline.shape[0], nseg + 1)     # 竖线沿 y 分段
    bh = np.linspace(0, hline.shape[1], nseg + 1)     # 横线沿 x 分段

    def build(line_img, orient, lines, s, bounds):
        m = len(lines)
        off = np.zeros((m, nseg))
        pk = np.zeros((m, nseg))
        ok = np.zeros((m, nseg), dtype=bool)
        for i, p in enumerate(lines):
            for k in range(nseg):
                d, v = _seg_measure(line_img, orient, p,
                                    bounds[k], bounds[k + 1] + 8, s)
                if v <= 0:
                    continue
                off[i, k], pk[i, k], ok[i, k] = d, v, True
        if ok.any():
            ref = float(np.median(pk[ok]))
            ok &= pk >= 0.45 * ref                 # 锁 1 显著性
        ok &= np.abs(off) <= lim_frac * s          # 锁 2 限幅
        return off, ok

    voff, vok = build(vline, "v", xs, sx, bv)
    hoff, hok = build(hline, "h", ys, sy, bh)
    voff = _fill_and_smooth(voff, vok, lim_frac * sx)   # 锁 3 平滑
    hoff = _fill_and_smooth(hoff, hok, lim_frac * sy)

    mids_v = (bv[:-1] + bv[1:]) / 2
    mids_h = (bh[:-1] + bh[1:]) / 2
    pts = [[None] * nx for _ in range(ny)]
    for i in range(nx):
        for j in range(ny):
            pts[j][i] = (float(xs[i] + np.interp(ys[j], mids_v, voff[i])),
                         float(ys[j] + np.interp(xs[i], mids_h, hoff[j])))
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


def classify_full(gray_cls, xs, ys, pts=None, need_cross=True, gap=0.70,
                  diag=False):
    """整盘专属分类：核心亮度主导 + 自适应阈值 + 交点校验。

    判据选择依据（实测 page1_b0/b1/b2，格距 17.8~21.6px）：
    ① 环判据（旧版）在 20px 格距下不可用：真白子 ring 0.13~0.39 与边缘伪影
       0.13~0.29 完全重叠，且环上暗点只占 1~2 个象限（不是圆环）——1px 描边
       经 JPEG + 缩放后已不存在。降门槛只会同时放进噪声，故整判据弃用。
    ② 核心亮度才是可分特征：交点处两条线交叉，空格核心暗（众数 75~95），
       白子是纸白（190~209）。但**必须抗偏移**：曲线网格的采样点仍有 ±2px
       抖动，固定中心窗会把采样点偏到纸面上，空格 core 从 127 一路拖尾到
       180，与白子粘成一片（实测连续单峰，无谷）。改用
       3x3 中位 → 5x5 最小值（等价于"窗口内最暗的 3x3"），容忍 ±2px 偏移，
       空格被压回 60~120，中间出现 40+ 灰阶的空隙。
    ③ 阈值自适应：thr = 空格众数 + gap*(纸面中位 - 空格众数)。
       纸色/墨色逐题不同（med 189~197、众数 75~95），固定阈值不可移植。
       gap=0.70 实测三题阈值 155/165/166，落在各题空隙内；
       gap 降到 0.55 会把拖尾空格吸进来（b1 白 24→29）。
    ④ 交点校验：core 只在采样点压在交点上时才可分，要求至少一条线可见。
    ⑤ 黑子判据不变（dark_frac 与 _big_dark_frac 双条件，实测 0.89~0.98）。

    diag=True 时第三返回值为逐格特征表 (ix, iy, label, dark_frac, core, hv)。
    """
    black, white, marks = set(), set(), []
    rows = [] if diag else None
    dx = float(np.median(np.diff(xs)))
    dy = float(np.median(np.diff(ys)))
    s = min(dx, dy)
    med = float(np.median(gray_cls))
    nx, ny = len(xs), len(ys)
    rd = max(int(s * 0.38), 5)
    r_line = int(s * 0.35)
    # 抗偏移核心亮度图：3x3 中位去噪 → 5x5 最小值
    core_map = cv2.erode(cv2.medianBlur(gray_cls, 3), np.ones((5, 5), np.uint8))
    cores, cand = [], []
    for ix in range(nx):
        for iy in range(ny):
            if pts is not None:
                fx, fy = pts[iy][ix]
            else:
                fx, fy = xs[ix], ys[iy]
            cx, cy = int(round(fx)), int(round(fy))
            cx = int(np.clip(cx, 0, gray_cls.shape[1] - 1))
            cy = int(np.clip(cy, 0, gray_cls.shape[0] - 1))
            disc = gray_cls[max(cy - rd, 0):cy + rd + 1,
                            max(cx - rd, 0):cx + rd + 1]
            dark_frac = float(np.mean(disc < 110)) if disc.size else 0.0
            if dark_frac > 0.5 and D._big_dark_frac(
                    gray_cls, cx, cy, max(int(s * 0.44), 6)) > 0.5:
                black.add((ix, iy))
                marks.append((cx, cy, r_line, "B"))
                if diag:
                    rows.append((ix, iy, "B", dark_frac, 0.0, 0.0))
                continue
            core = float(core_map[cy, cx])
            cores.append(core)
            cand.append((ix, iy, cx, cy, core, dark_frac))
    # 自适应阈值：空格众数 → 纸面中位 之间按 gap 取点
    if cores:
        h, e = np.histogram(np.asarray(cores),
                            bins=np.arange(60, max(min(med, 250), 70), 10))
        mode = float(e[int(np.argmax(h))] + 5) if len(h) else float(np.median(cores))
        thr = mode + gap * (med - mode)
    else:
        thr = med - 10.0
    for (ix, iy, cx, cy, core, dark_frac) in cand:
        hv = 1.0
        if need_cross:
            hf, vf = D._line_visibility(gray_cls, cx, cy, r_line,
                                        ix == 0, ix == nx - 1,
                                        iy == 0, iy == ny - 1)
            hv = max(hf, vf)
            if hv < 0.5:
                if diag:
                    rows.append((ix, iy, ".", dark_frac, core, hv))
                continue
        is_white = core >= thr
        if is_white:
            white.add((ix, iy))
            marks.append((cx, cy, r_line, "W"))
        if diag:
            rows.append((ix, iy, "W" if is_white else ".", dark_frac, core, hv))
    return black, white, marks if not diag else (black, white, rows)
