"""棋形识别：纸质死活题照片 -> 网格坐标。

经验（来自 SKILL.md 校对清单，勿回退）：
- 不用投影法找棋盘线（棋子主导投影出假线）——用 HoughLinesP 检测印刷线 + 圆心聚类兜底
- 不用交叉点暗像素分类（白子细边与空点线十字不可分）——用 HoughCircles + 圆心亮度
- HoughCircles param2=28 紧凑可靠；param2=25 会把蓝字/印刷字误检为白子
- 产出叠加核对图（_grid.png），人工核对每颗子是必须质检步骤
- 手写答案双人复核：识别出的编号序列必须列给用户确认，读不准标 "?"
"""
import math

import cv2
import numpy as np


def _cluster(vals, gap):
    """把一组坐标值按间距 gap 聚类，返回各类中心。"""
    vals = sorted(vals)
    groups, cur = [], [vals[0]]
    for v in vals[1:]:
        if v - cur[-1] <= gap:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def _detect_lines(gray):
    """HoughLinesP 检印刷棋盘线 -> (line_xs, line_ys)。容忍±8°倾斜。"""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=w // 6, maxLineGap=8)
    line_xs, line_ys = [], []
    if lines is not None:
        for l in lines.reshape(-1, 4):  # cv2 5.x 返回 (N,4)，旧版 (N,1,4)
            x1, y1, x2, y2 = map(float, l)
            dx_, dy_ = abs(x2 - x1), abs(y2 - y1)
            if dy_ < 0.15 * dx_ and dx_ > w // 8:
                # 贴图边的全线是页边/照片边界伪影（同 _detect_raw_lines）
                if min(y1, y2) > 6 and max(y1, y2) < h - 6:
                    line_ys.append((y1 + y2) / 2)
            elif dx_ < 0.15 * dy_ and dy_ > h // 8:
                if min(x1, x2) > 6 and max(x1, x2) < w - 6:
                    line_xs.append((x1 + x2) / 2)
    return line_xs, line_ys


def _rough_circles(gray):
    """粗检圆（严参数 param2=28，只作网格融合的种子）。
    必须用严参数：松参数会把四子拐角处的圆弧组合检成"幽灵圆"，
    幽灵圆心落在半间距位置，会带歪网格间距（实测 s=44 被当成 22）。"""
    h, w = gray.shape
    c = cv2.HoughCircles(cv2.GaussianBlur(gray, (5, 5), 1), cv2.HOUGH_GRADIENT,
                         dp=1.2, minDist=18, param1=80, param2=28,
                         minRadius=max(6, w // 60), maxRadius=max(20, w // 10))
    if c is None:
        return []
    pts = [(float(p[0]), float(p[1])) for p in c[0]]
    # 去重：同一子的同心孪生检测（距<18px）会污染间距先验（造出 14.5 的假间距）
    out = []
    for p in pts:
        for q in out:
            if (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 < 18 ** 2:
                break
        else:
            out.append(p)
    return out


def _fuse(line_pos, center_pos):
    """线位置 + 圆心双证据融合出等距网格。失败返回 None。

    间距估计用「整数倍得分」稳健法：相邻类中心差值大多是真实间距 s 的
    整数倍，候选 s 取各差值，选满足 |d/s - round(d/s)|<0.15 最多的；
    同分取较大 s——棋子圆弧的切线段会造出半间距假线（实测同一行相邻
    子的弧段被 maxLineGap 连成 200px 长线），取最小差值会中招。"""
    vals = sorted(line_pos + center_pos)
    if len(vals) < 2:
        return None
    centers_c = _cluster(vals, 6)
    if len(centers_c) < 2:
        return None
    diffs = [b - a for a, b in zip(centers_c, centers_c[1:]) if b - a >= 8]
    if not diffs:
        return None
    best_s, best_score = None, -1
    for s in sorted(set(diffs)):
        score = sum(1 for d in diffs
                    if round(d / s) >= 1 and abs(d / s - round(d / s)) < 0.15)
        if score > best_score or (score == best_score and best_s and s > best_s):
            best_s, best_score = s, score
    s = float(best_s)
    grid, cur = [], centers_c[0]
    while cur <= centers_c[-1] + s * 0.5:
        near = [c for c in centers_c if abs(c - cur) < s * 0.35]
        grid.append(float(np.mean(near)) if near else cur)
        cur = grid[-1] + s
    return grid if len(grid) >= 2 else None


def _ring_dark_frac(gray, cx, cy, r, n=36):
    """圆周上暗像素占比。真白子有印刷描边黑环，假圆（印刷字/噪声）没有。"""
    h, w = gray.shape
    vals = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        x, y = int(cx + r * np.cos(a)), int(cy + r * np.sin(a))
        if 0 <= x < w and 0 <= y < h:
            vals.append(gray[y, x])
    return float(np.mean([v < 150 for v in vals])) if vals else 0.0


def _ring_outline_frac(gray, cx, cy, r, n=72):
    """白子描边环检测：圆周暗像素占比，但剔除横竖线穿环的 4 个角区（±14°）。

    为什么需要：部分题册是「线穿白子」印刷（白子描边空心、网格线从子身上
    穿过），此时"线可见=空点"判据全灭（实测白子 hf 0.6-0.9）。描边是完整
    圆环，剔除线交叉角区后仍高（实测 0.15-0.44）；空点的环上暗像素全部
    来自线交叉（含粗边线/角点，未剔除时高达 0.53），剔除后≈0。
    暗阈值 120 而非 150：板外纸面/桌面阴影灰（~145-151）会被 150 当成暗，
    底边整排误报白子（实测环占比 0.31-0.40）；描边墨色核心 <120。"""
    h, w = gray.shape
    vals = []
    for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
        deg = math.degrees(a) % 90
        if deg < 14 or deg > 76:
            continue
        x, y = int(cx + r * np.cos(a)), int(cy + r * np.sin(a))
        if 0 <= x < w and 0 <= y < h:
            vals.append(gray[y, x])
    return float(np.mean([v < 120 for v in vals])) if vals else 0.0


def _spacing_from_lines(gray):
    """从印刷线位置直接估间距——棋盘最干净、最密集的证据。
    棋子圆心先验在子少时会退化（实测一图多题切出的裁剪图只有 2 颗黑子，
    prior 返回 None，回退到全体圆心后被幽灵圆带成半间距 21.5，网格翻倍）。
    用 _detect_raw_lines（宽松参数，线段多）而非 _detect_lines（严格，
    稀疏棋盘丢线太多会把间距估成 2 倍）。"""
    segs = _detect_raw_lines(gray)

    def est(vals):
        if len(vals) < 3:
            return None
        c = _cluster(sorted(vals), 12)
        if len(c) < 2:
            return None
        d = [b - a for a, b in zip(c, c[1:]) if 20 <= b - a <= 250]
        if not d:
            return None
        med = float(np.median(d))
        # 一致性：多数相邻差应接近中位数，否则不是规则网格
        n_ok = sum(1 for x in d if abs(x - med) < 0.35 * med)
        if n_ok < max(1, (len(d) + 1) // 2):
            return None
        return med

    sx = est([(s[0] + s[2]) / 2 for s in segs if s[4] == "v"])
    sy = est([(s[1] + s[3]) / 2 for s in segs if s[4] == "h"])
    if sx and sy:
        # 线缺失只会让估计偏大（跳过缺线），不会偏小——取小的
        return min(sx, sy)
    return sx or sy


def _spacing_score(reps, s):
    """候选间距 s 对线位置证据的拟合得分（供线估计与圆心先验冲突时仲裁）。
    最佳相位下：匹配的线位置数 + 匹配的节点数 - 节点总数。
    幽灵半间距会在真线之间造出一倍空节点被重罚；线缺失时 2s 会漏掉
    一半真实线位置，命中数上不去。真间距两边都满分。"""
    if len(reps) < 2:
        return -1e9
    lo, hi = reps[0], reps[-1]
    tol = max(2.5, 0.2 * s)
    best = -1e9
    a = lo - s * 0.4
    while a <= lo + s * 0.4:
        nodes = []
        x = a
        while x <= hi + s * 0.25:
            if x >= lo - s * 0.25:
                nodes.append(x)
            x += s
        if nodes:
            matched_nodes = sum(
                1 for n in nodes if any(abs(r - n) < tol for r in reps))
            matched_reps = sum(
                1 for r in reps if any(abs(r - n) < tol for n in nodes))
            best = max(best, matched_reps + matched_nodes - len(nodes))
        a += 1.0
    return best


def _spacing_prior(centers):
    """圆心两两距离 -> 间距先验（细粒度整数倍得分）。
    不用直方图：44±2 的距离会被 bin 边界劈半，输给 62(=44√2 对角距)。
    对候选 s（0.5 步长）统计满足 |d/s-round(d/s)|<0.08 的距离数，取最高。
    >=3 个圆才可信。"""
    if len(centers) < 3:
        return None
    pts = np.array(centers)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    d = d[np.triu_indices(len(pts), 1)]
    d = d[(d >= 12) & (d <= 300)]
    if len(d) == 0:
        return None
    best_s, best_score = None, -1
    s = 20.0  # 书页题图在常规分辨率下格子≥25px，<20 的"间距"必是伪影
    while s <= 120:
        score = 0
        for v in d:
            k = round(v / s)
            # 绝对容差：相对容差会让 √5 距离(98)混进 s=95（88 误差 7px 也算"≈1倍"）
            if k >= 1 and abs(v - k * s) < max(2.5, 0.05 * k * s):
                score += 1
        # 同分取大 s：半间距是任何距离的"公约数"，平票必然赢，必须压制
        if score > best_score or (score == best_score and best_s and s > best_s):
            best_s, best_score = s, score
        s += 0.5
    return best_s


def _fit_lattice(vals, s):
    """给定间距 s，在证据位置上拟合等距点阵：扫相位 anchor，
    取命中数最多者（命中=距点阵节点 <0.22s），节点取命中均值。
    幽灵圆心落在半间距位置，不参与命中，自然被排除。"""
    vals = sorted(vals)
    lo, hi = vals[0], vals[-1]
    best_a, best_hits = None, []
    a = lo - s * 0.4
    while a <= lo + s * 0.4:
        hits = []
        for v in vals:
            r = (v - a) % s
            if r < s * 0.22 or r > s * 0.78:
                hits.append(v)
        if len(hits) > len(best_hits):
            best_a, best_hits = a, hits
        a += 0.5
    if len(best_hits) < 2:
        return None
    grid = []
    k = int(np.floor((lo - best_a) / s))
    while best_a + k * s <= hi + s * 0.4:
        node = best_a + k * s
        # 只保留证据范围内的节点：范围外的外推节点没有证据支撑，
        # 会在棋盘边上造出幻影行/列（实测顶部多出 y=-7 的空行）
        if lo - s * 0.3 <= node <= hi + s * 0.3:
            near = [v for v in best_hits if abs(v - node) < s * 0.25]
            grid.append(float(np.mean(near)) if near else node)
        k += 1
    return grid if len(grid) >= 2 else None


def _fit_lattice2(phase_vals, range_vals, s):
    """相位/范围分离的点阵拟合（_fit_lattice 的抗幽灵版）。

    模糊低对比照片上白子幽灵圆成群且相位一致（摞子边缘弧，-0.47s），
    实测投票精检 46 颗中 29 颗幽灵、真白子仅 10 颗——幽灵占多数时
    _fit_lattice 按命中数扫相位会被"多数暴政"带歪半格。
    黑子圆心过 <110 亮度验证几乎零幽灵，故相位只信黑子圆心+印刷线；
    网格范围与节点精化仍用全部证据（含白子圆心，补无黑子的行/列）。
    相位证据不足 2 个时退回全体证据（等效旧 _fit_lattice）。"""
    if len(phase_vals) < 2:
        phase_vals = range_vals
    if len(range_vals) < 2 or s < 8:
        return None
    phase_vals = sorted(phase_vals)
    range_vals = sorted(range_vals)
    best_a, best_hits = None, -1
    lo_p = phase_vals[0]
    a = lo_p - s * 0.4
    while a <= lo_p + s * 0.4:
        hits = 0
        for v in phase_vals:
            r = (v - a) % s
            if r < s * 0.22 or r > s * 0.78:
                hits += 1
        if hits > best_hits:
            best_a, best_hits = a, hits
        a += 0.5
    if best_hits < 2:
        return None
    lo, hi = range_vals[0], range_vals[-1]
    grid = []
    k = int(np.floor((lo - best_a) / s))
    while best_a + k * s <= hi + s * 0.4:
        node = best_a + k * s
        if lo - s * 0.3 <= node <= hi + s * 0.3:
            # 节点精化只用相位证据（线+黑子圆心）：白子幽灵圆成片落在
            # 真节点 -0.3s 处（恰在 0.25s 精化窗内），参与取均值会把节点
            # 拉偏 20-30px，扫描分类在空点上踩到子缘/笔迹出幻影白子
            near = [v for v in phase_vals if abs(v - node) < s * 0.25]
            grid.append(float(np.mean(near)) if near else node)
        k += 1
    return grid if len(grid) >= 2 else None


def _cross_dark(gray, cx, cy, r):
    """过圆心的横竖两条线上暗像素数。
    真白子的中心十字被子体盖住 ≈0；空交叉点幻影有网格线十字穿过 ≥10。"""
    h, w = gray.shape
    cx, cy, r = int(cx), int(cy), int(r)
    vals = []
    for x in range(max(cx - r, 0), min(cx + r, w)):
        vals.append(gray[cy, x])
    for y in range(max(cy - r, 0), min(cy + r, h)):
        vals.append(gray[y, cx])
    return int(sum(1 for v in vals if v < 150))


def _consensus_circles(gray, s):
    """多 param2 投票检圆：28/24/20 三跑，位置聚类计票。
    黑子：圆心暗（<110）即强证据，1 票也收；
    白子（纸面低对比，最易漏也最易假）：≥2 票或 1 票+描边环>0.3，
    且必须过中心十字检验（排除棋盘边上"子弧+边线"围成的幻影圆）。"""
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    votes = []  # [cx, cy, n]
    for p2 in (28, 24, 20):
        raw = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=int(s * 0.65), param1=80, param2=p2,
                               minRadius=int(s * 0.30), maxRadius=int(s * 0.58))
        for p in (raw[0] if raw is not None else []):
            cx, cy = float(p[0]), float(p[1])
            for v in votes:
                if (cx - v[0]) ** 2 + (cy - v[1]) ** 2 < (s * 0.3) ** 2:
                    v[0] = (v[0] * v[2] + cx) / (v[2] + 1)
                    v[1] = (v[1] * v[2] + cy) / (v[2] + 1)
                    v[2] += 1
                    break
            else:
                votes.append([cx, cy, 1])
    out = []
    for cx, cy, n in votes:
        m = int(gray[max(int(cy) - 3, 0):int(cy) + 4,
                     max(int(cx) - 3, 0):int(cx) + 4].mean())
        if m < 110:
            out.append((cx, cy))  # 黑子：亮度即验证
        elif n >= 2 or _ring_dark_frac(gray, cx, cy, s * 0.44) > 0.3:
            if _cross_dark(gray, cx, cy, s * 0.35) <= 8:
                out.append((cx, cy))  # 白子：投票/描边环 + 十字检验
    return out


def _cluster_gapfill(line_pos, s):
    """印刷线位置聚类 + 缺口插值出网格（detect_grid 主路径）。

    替代等距点阵拟合：真实照片行距带透视渐变（实测一张书页顶到底
    61.5→71.8px），单一间距的点阵在远端累积漂移超过 0.22s 容差，
    会丢线、错相位；且圆心证据幽灵多（手写笔迹、印刷装饰都会出假圆）。
    这里只信线证据：相邻簇间距按 s 的整数倍插值补线（被棋子压断的线，
    实测中间连续缺 2 条）；间距结构对不上就整体返回 None 走回退路径，
    不硬猜。"""
    if len(line_pos) < 3:
        return None
    c = _cluster(sorted(line_pos), 12)
    if len(c) < 3:
        return None
    out = [c[0]]
    for a, b in zip(c, c[1:]):
        gap = b - a
        k = int(round(gap / s))
        if k < 2 and abs(gap - s) >= 0.4 * s:
            # 相邻簇间距既不是 ~s 也不是 s 的整数倍：线证据被污染。
            # 实测模糊低对比照片上，同列摞子的左缘切线连成"幽灵竖线"，
            # 位置在真线 -0.47s 处，与真线差 0.57s 被当成相邻线，
            # 整盘相位错半格、扫描分类全崩（全判白）。结构不可信，
            # 返回 None 走圆心证据回退路径，不硬猜。
            return None
        if k >= 2:
            if k > 4 or abs(gap / k - s) >= 0.3 * s:
                return None
            for i in range(1, k):
                out.append(a + gap * i / k)
        out.append(b)
    return out if len(out) >= 3 else None


def _grid_hits_circles(xs, ys, circles, min_frac=0.6):
    """网格与圆心证据一致性校验：圆心落在节点 0.28s 内的占比。

    线证据被幽灵线整体带偏半格时（摞子边缘切线间距恰好也 ≈s，
    _cluster_gapfill 的结构校验拦不住），圆心是唯一独立证据。
    粗检圆（param2=28 严参数）里幽灵占比低，真子必然压节点。
    圆太少（<3）无法裁决时放行，交给后续画回确认门禁。"""
    if len(circles) < 3:
        return True
    sx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1e9
    sy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1e9
    if sx < 8 or sy < 8:
        return True
    hits = 0
    for cx, cy in circles:
        dx = min(abs(cx - x) for x in xs)
        dy = min(abs(cy - y) for y in ys)
        if dx < 0.28 * sx and dy < 0.28 * sy:
            hits += 1
    return hits / len(circles) >= min_frac


def detect_grid(gray):
    """完整网格检测。返回 (xs, ys, best_circles)。

    管线：粗检圆 → 亮度分流（黑子中心<110 几乎零误判）→ 仅黑子圆心
    出间距先验（幽灵圆全是亮色，进不来；实测四子拐角幽灵会把 44 带歪成
    22/62/15.5）→ 点阵相位拟合（容忍离群证据）→ 投票精检圆 → 联合再拟合。"""
    line_xs, line_ys = _detect_lines(gray)
    c1 = _rough_circles(gray)

    def brightness(cx, cy):
        return int(gray[max(int(cy) - 3, 0):int(cy) + 4,
                        max(int(cx) - 3, 0):int(cx) + 4].mean())

    blacks = [c for c in c1 if brightness(*c) < 110]
    # 间距决策：线位置估计优先（干净、不受棋子数量/蓝字影响），圆心先验兜底
    # （线被大面积压断时）。两者冲突（一个是另一个约 2 倍——幽灵半间距 vs
    # 线缺失 2s）时，用线位置证据打分仲裁（_spacing_score）。
    s_line = _spacing_from_lines(gray)
    s_circ = _spacing_prior(blacks) or _spacing_prior(c1)
    s0 = None
    if s_line and s_circ and max(s_line, s_circ) / min(s_line, s_circ) > 1.55:
        segs = _detect_raw_lines(gray)
        reps_x = _cluster(sorted((t[0] + t[2]) / 2 for t in segs if t[4] == "v"), 12)
        reps_y = _cluster(sorted((t[1] + t[3]) / 2 for t in segs if t[4] == "h"), 12)
        cands = {s_line: max(_spacing_score(reps_x, s_line),
                             _spacing_score(reps_y, s_line)),
                 s_circ: max(_spacing_score(reps_x, s_circ),
                             _spacing_score(reps_y, s_circ))}
        s0 = max(cands, key=cands.get)
    else:
        s0 = s_line or s_circ
    xs = ys = None
    if s0:
        # 主路径：纯线证据聚类+插值（免疫透视渐变与幽灵圆）
        xs = _cluster_gapfill(line_xs, s0)
        ys = _cluster_gapfill(line_ys, s0)
        if xs and ys:
            # 线网格一致性校验只用黑子圆心（<110 亮度验证，几乎零幽灵）：
            # 粗检圆里白子幽灵成群（实测 37 颗里 26 颗幽灵），用全体圆心
            # 会把正确网格误判成不一致（真子 11/37=0.30 < 0.6 阈值）
            if not _grid_hits_circles(xs, ys, blacks):
                # 线网格与黑子证据矛盾（幽灵线整体半格偏移），信圆心，走回退
                xs = ys = None
    if not xs or not ys:
        # 回退路径：线太少时靠圆心证据的点阵/融合拟合（勿动，老路兜底）
        xs = ys = None
        c2 = []
        if s0 and s0 >= 8:
            # 模糊/低对比照片上粗检圆幽灵成群（实测一张 194 颗、真子仅 17），
            # 粗圆心直接进点阵拟合会把相位带歪半格；先投票精检提纯再拟合。
            # 提纯不足 4 颗时退回粗圆心（裁剪小子图常见），宁滥勿缺。
            c2 = _consensus_circles(gray, s0)
        ev = c2 if len(c2) >= 4 else c1
        ev_black = [c for c in ev if brightness(*c) < 110]
        if s0:
            xs = _fit_lattice2(line_xs + [c[0] for c in ev_black],
                               line_xs + [c[0] for c in ev], s0)
            ys = _fit_lattice2(line_ys + [c[1] for c in ev_black],
                               line_ys + [c[1] for c in ev], s0)
        if not xs or not ys:
            xs = _fuse(line_xs, [c[0] for c in ev])
            ys = _fuse(line_ys, [c[1] for c in ev])
        if not xs or not ys:
            return None
        s = min(np.median(np.diff(xs)) if len(xs) > 1 else 1e9,
                np.median(np.diff(ys)) if len(ys) > 1 else 1e9)
        if s == 1e9 or s < 8:
            return xs, ys, c1
        if not c2:
            c2 = _consensus_circles(gray, s)
        if c2:
            # 精检圆心 + 线联合证据再拟合：补被棋子压断的线。
            # 同样走相位/范围分离（防幽灵多数暴政），且须过黑子一致性校验
            c2b = [c for c in c2 if brightness(*c) < 110]
            xs2 = _fit_lattice2(line_xs + [c[0] for c in c2b],
                                line_xs + [c[0] for c in c2], s)
            ys2 = _fit_lattice2(line_ys + [c[1] for c in c2b],
                                line_ys + [c[1] for c in c2], s)
            if xs2 and ys2 and _grid_hits_circles(xs2, ys2, c2b):
                if len(xs2) >= len(xs):
                    xs = xs2
                if len(ys2) >= len(ys):
                    ys = ys2
            return xs, ys, c2
    return xs, ys, c1


def grid_from_stones(centers):
    """线检测失败时，用棋子圆心坐标聚类出网格（兜底）。"""
    xs = _cluster([c[0] for c in centers], 12)
    ys = _cluster([c[1] for c in centers], 12)
    return (xs, ys) if len(xs) >= 2 and len(ys) >= 2 else None


def _snap_grid_to_lines(gray, xs, ys, circles):
    """网格线吸附：把每根网格线精化到暗像素投影峰（±0.2s 搜索窗）。

    圆心锚定的点阵有系统性偏移：锚点是石子中心（边线处的子半悬在外，
    中心偏出线外 7px 实测），s 估计差 0.6% 累积到远端共 -15~-29px。
    偏差 >10px 时扫描分类的 ±10px 采样带套不住真线，邻子描边环进带
    被当成"线可见"，边点白子误判成空点（实测 3 处全因此）。
    对每根线在 ±0.2s 内找暗像素覆盖率最高的位置；石子邻域（覆盖线段的
    部分）从统计中剔除，避免子体把峰拉向自己。证据不足（<0.15）保持原位。
    """
    h, w = gray.shape
    dark = (gray < 150).astype(np.uint8)

    def stone_free_mask(length, positions, s):
        """positions: 该方向上的石子中心坐标；返回可统计位置的布尔数组。"""
        ok = np.ones(length, dtype=bool)
        for p in positions:
            lo = max(int(p - s * 0.5), 0)
            hi = min(int(p + s * 0.5), length)
            ok[lo:hi] = False
        return ok

    def snap(pos, orient, lo, hi, stones, s):
        band = int(s * 0.2)
        half = max(int(s * 0.02), 2)  # 线厚 ±2px

        def coverage(cand):
            if orient == "h":
                y0, y1 = max(cand - half, 0), min(cand + half + 1, h)
                strip = dark[y0:y1, max(int(lo), 0):min(int(hi), w)]
                if strip.size == 0:
                    return None
                cov_line = strip.any(axis=0)
            else:
                x0, x1 = max(cand - half, 0), min(cand + half + 1, w)
                strip = dark[max(int(lo), 0):min(int(hi), h), x0:x1]
                if strip.size == 0:
                    return None
                cov_line = strip.any(axis=1)
            free = stone_free_mask(len(cov_line), stones, s)
            if free.sum() < 8:
                return None
            return float(cov_line[free].mean())

        best_pos, best_cov = None, 0.0
        for cand in range(int(pos) - band, int(pos) + band + 1, 2):
            cov = coverage(cand)
            if cov is not None and cov > best_cov:
                best_cov, best_pos = cov, float(cand)
        # 只在显著优于原位时才移动（+0.12）：原位证据已经不错时，
        # 邻子/笔迹造成的次峰不值得追（实测密子区会把线吸附到子行上）
        cur_cov = coverage(int(pos)) or 0.0
        if best_pos is not None and best_cov > max(0.15, cur_cov + 0.12):
            return best_pos
        return pos

    sx = float(np.median(np.diff(xs))) if len(xs) > 1 else 20
    sy = float(np.median(np.diff(ys))) if len(ys) > 1 else 20
    # 每根线只剔除压在该线上的石子邻域（|石坐标-线位|<0.5s），
    # 方向别搞反：竖线的覆盖统计沿 y 走，剔除的是石子的 y 坐标
    xs2 = [snap(x, "v", ys[0], ys[-1],
                [c[1] for c in circles if abs(c[0] - x) < sx * 0.5], sx)
           for x in xs]
    ys2 = [snap(y, "h", xs2[0], xs2[-1],
                [c[0] for c in circles if abs(c[1] - y) < sy * 0.5], sy)
           for y in ys]

    def uniform(vals, s):
        """吸附后间距仍须大致均匀：任何间距 <0.6 或 >1.5 倍中位数即失败。"""
        d = np.diff(vals)
        m = float(np.median(d))
        return bool(np.all(d > 0.6 * m) and np.all(d < 1.5 * m))

    # 均匀性校验：吸附把网格拉散（密子区次峰）就整轴回退，宁要粗网格
    if len(xs2) > 2 and not uniform(xs2, sx):
        xs2 = list(xs)
    if len(ys2) > 2 and not uniform(ys2, sy):
        ys2 = list(ys)
    return xs2, ys2


def _line_visibility(gray, cx, cy, r, edge_l, edge_r, edge_t, edge_b, ignore=None):
    """网格线在该节点是否可见（印刷图"有子必盖线"）。返回 (横线占比, 竖线占比)。

    三个关键设计，全是踩坑换来的：
    - 粗采样带 ±10px 再投影成 1D：节点拟合常有偏差（角点证据少、透视下
      实测达 6.5-9px），窄带采样会整条线漏掉（实测角点 cross_dark=4 vs 中部 34）
    - 边缘/角节点只采棋盘内侧半窗：边节点外侧本来就没有线，
      全窗采样会让暗像素天然减半，固定阈值必然把边缘空点误判成白子
    - 按采样长度归一化成占比而非绝对像素数：间距 s 变化时绝对数不可比
    - ignore（蓝字掩码）：蓝字像素从分子分母同时剔除。写"涂白"不行——
      数字写在棋子上是正常用法，涂白会吃掉黑子中心；抗锯齿残边又会让
      白子上的线显得可见"""
    h, w = gray.shape
    cx, cy, r = int(cx), int(cy), int(r)
    # 带宽 ±10px：透视下节点拟合偏差实测达 6.5-9px，窄带会整条线漏掉。
    # 上限推导：白子半径 ~0.44s，描边环要进带需 |dy|<=band 且 |dx|<=r，
    # band=11 时环点 dx=sqrt((0.44s)^2-band^2)>r=0.35s 刚好进不来，band>=12 才中招
    band = 10
    x0 = cx if edge_l else max(cx - r, 0)
    x1 = cx if edge_r else min(cx + r, w - 1)
    if x1 <= x0:
        x0, x1 = max(cx - r, 0), min(cx + r, w - 1)
    rows = gray[max(cy - band, 0):cy + band + 1, x0:x1 + 1]
    ig = None
    if ignore is not None:
        ig = ignore[max(cy - band, 0):cy + band + 1, x0:x1 + 1] > 0
    if rows.size == 0:
        hf = 0.0
    elif ig is not None:
        darkp = (rows < 150) & (~ig)
        validcol = (~ig).any(axis=0)
        hf = float(darkp.any(axis=0)[validcol].mean()) if validcol.any() else 0.0
    else:
        hf = float(np.mean((rows < 150).any(axis=0)))
    y0 = cy if edge_t else max(cy - r, 0)
    y1 = cy if edge_b else min(cy + r, h - 1)
    if y1 <= y0:
        y0, y1 = max(cy - r, 0), min(cy + r, h - 1)
    cols = gray[y0:y1 + 1, max(cx - band, 0):cx + band + 1]
    if ignore is not None:
        igv = ignore[y0:y1 + 1, max(cx - band, 0):cx + band + 1] > 0
    else:
        igv = None
    if cols.size == 0:
        vf = 0.0
    elif igv is not None:
        darkp = (cols < 150) & (~igv)
        validrow = (~igv).any(axis=1)
        vf = float(darkp.any(axis=1)[validrow].mean()) if validrow.any() else 0.0
    else:
        vf = float(np.mean((cols < 150).any(axis=1)))
    return hf, vf


def _ignore_mask(bgr):
    """手写蓝字掩码（膨胀 2 轮盖住抗锯齿边缘），供棋子分类剔除用。"""
    return cv2.dilate(_blue_mask(bgr), np.ones((3, 3), np.uint8), iterations=2)


def _big_dark_frac(gray, cx, cy, half):
    """交点窗口内「含中心的暗连通域」面积占比（黑子专属特征）。

    黑子 = 0.44s 的实心大暗盘（占比 0.63~0.85）；网格线交叉/墙面线是细长条、
    印刷小暗斑只有 20~30px，占比 <=0.45；空点 <=0.3。
    为什么不用 dark_frac（盘内暗像素比）：底行交点压在 14px 粗墙线上时
    dark_frac 可达 0.51 被误判黑子，而墙线细长的连通域占比只有 0.45。
    """
    H, W = gray.shape
    x0, x1 = max(cx - half, 0), min(cx + half + 1, W)
    y0, y1 = max(cy - half, 0), min(cy + half + 1, H)
    win = gray[y0:y1, x0:x1] < 110
    if not win.any():
        return 0.0
    n, lab = cv2.connectedComponents(win.astype(np.uint8), connectivity=8)
    lid = lab[cy - y0, cx - x0] if (0 <= cy - y0 < lab.shape[0]
                                    and 0 <= cx - x0 < lab.shape[1]) else 0
    if lid:
        return float((lab == lid).sum()) / win.size
    sizes = [(lab == i).sum() for i in range(1, n)]
    return (max(sizes) / win.size) if sizes else 0.0


def _core_bright(gray, cx, cy, s):
    """白子内部亮度：取半径 0.16s~0.34s 环带的中位，并剔除四个轴向 ±22° 扇区。

    为什么不直接用中心小窗：题册是「线穿白子」印刷，网格线正好从子心穿过，
    中心窗会整片压在线上，实测真白子 core-med 掉到 -17~-19（warp 后格距恒为
    44，中心窗只有 7x7），比空点还暗，白子判据被自己的线判死。
    环带 + 剔轴向后取到的才是真正的纸色核心。
    """
    h, w = gray.shape
    r_in, r_out = 0.16 * s, 0.34 * s
    n = int(r_out) + 1
    xs_, ys_, vs = [], [], []
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            r = (dx * dx + dy * dy) ** 0.5
            if not (r_in <= r <= r_out):
                continue
            deg = math.degrees(math.atan2(abs(dy), abs(dx)))   # 0=水平 90=垂直
            if deg < 22 or deg > 68:      # 剔除横竖线穿过的四个扇区
                continue
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                vs.append(gray[y, x])
    return float(np.median(vs)) if vs else float(gray[cy, cx])


def classify_by_sweep(gray, xs, ys, ignore=None):
    """全交叉点扫描分类（印刷题图专用，比 HoughCircles 稳一个量级）：
    - 中心圆盘暗像素占比 >0.5 → 黑子（比单点亮度抗噪：空点中心压线也偏暗，
      实测空点中心亮度 116，距旧阈值 110 仅一线之隔，但圆盘占比仅 ~0.3）
    - 否则横竖任一线可见（占比 >0.4）→ 空点
    - 否则 → 白子（网格线被白子盖住了）
    依据：印刷图里"有子必盖线"。真实木盘照片不适用（木纹/透视），勿挪用。
    ignore：蓝字掩码。数字写在棋子上是正常用法（写在哪一手上），蓝字像素
    从分子分母同时剔除——涂白会把黑子中心吃掉、残边让白子上的线假可见。"""
    dx = np.median(np.diff(xs)) if len(xs) > 1 else 20
    dy = np.median(np.diff(ys)) if len(ys) > 1 else 20
    med = float(np.median(gray))   # 纸面参考亮度（白子核心应接近此值）
    r = int(min(dx, dy) * 0.35)
    nx, ny = len(xs), len(ys)
    black, white, marks = set(), set(), []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            cx, cy = int(x), int(y)
            # 采样盘半径 0.38s：接近石子真实半径(0.44s)但不碰邻子(s 间距)。
            # 不能太小——数字写在棋子上是正常用法，小盘会被数字整个盖住
            # （分母不足，实测 15px 盘被字宽 22px 的"1"全覆盖）；也不能
            # ≥0.42s，会蹭到邻子边缘。空点的线占比随盘增大反而降低。
            rd = max(int(min(dx, dy) * 0.38), 5)
            sy0, sy1 = max(cy - rd, 0), cy + rd + 1
            sx0, sx1 = max(cx - rd, 0), cx + rd + 1
            disc = gray[sy0:sy1, sx0:sx1]
            if ignore is not None:
                ig = ignore[sy0:sy1, sx0:sx1] > 0
                valid = ~ig
                denom = int(valid.sum())
                # 分母太小（数字完全盖住采样盘）时保守判空，交给步骤②门禁
                dark_frac = (float((disc < 110) [valid].sum()) / denom
                             if denom >= 9 else 0.0)
            else:
                dark_frac = float(np.mean(disc < 110)) if disc.size else 0.0
            if dark_frac > 0.5 and _big_dark_frac(
                    gray, cx, cy, max(int(min(dx, dy) * 0.44), 6)) > 0.5:
                # 双条件：盘内暗占比高 **且** 暗的是一个大连通域。
                # 只用 dark_frac 会在底行/边行误判：交点压着 14px 粗墙线时
                # dark_frac 可到 0.51，但那是细长条不是黑子（2026-09-09 q4）
                black.add((ix, iy))
                marks.append((cx, cy, r, "B"))
                continue
            hf, vf = _line_visibility(gray, cx, cy, r,
                                      ix == 0, ix == nx - 1,
                                      iy == 0, iy == ny - 1, ignore=ignore)
            if max(hf, vf) <= 0.4:
                # 线被盖住 → 白子（传统印刷：白子盖线）
                white.add((ix, iy))
                marks.append((cx, cy, r, "W"))
            else:
                # 线"可见"不一定是空点：线穿白子印刷风格的白子 hf 0.6-0.9。
                # 用描边圆环仲裁：白子环上有完整墨环，空点（含粗线/阴影灰）
                # 环上≈0。双条件：强环直接判白；弱环要求至少一个方向线被
                # 明显盖住（真白子 min(hf,vf) 0.29-0.39，粗线/阴影误报是 1.0）
                ring = _ring_outline_frac(gray, cx, cy,
                                          max(int(min(dx, dy) * 0.44), 6))
                # 白子判据：描边环 + 亮核心（本册是「线穿白子」印刷风格，
                # 实测大多数真白子 hf/vf=1.0，"盖线"条件会漏掉全部白子）。
                # ③ 核心亮度 >= 纸面中位-15：白子=纸色+描边，核心亮；
                #    假 ring 的暗斑/阴影核心是暗的。必须用小中心窗，
                #    不能用 0.38s 采样盘中位——盘内纸面占多数，中位永远是纸面亮度。
                # 主判据仍用中心小窗：实测它对真实照片最稳（q4 只残留 1 颗幻影；
                # 全程改用环带采样会让铅笔涂写一整片过关，幻影涨到 5 颗）。
                ch = max(2, int(round(min(dx, dy) * 0.064)))
                core = float(np.median(gray[max(cy - ch, 0):cy + ch + 1,
                                            max(cx - ch, 0):cx + ch + 1]))
                # 试过两种放宽，都被真实照片否决，留档免得再走一遍：
                # ① 全程改用环带核心（_core_bright，避开穿子线）：q4 铅笔涂写
                #    一整片过关，幻影从 1 颗涨到 5 颗。
                # ② 强环(>0.5/0.7)时兜底用环带核心：合成图能过，但 q4 幻影
                #    涨到 3 颗，且涂写假环的 ring 比真白子还高，分不开。
                # 最终只留中心窗判据——真实四题里只有 q4 残留 1 颗幻影 + 1 颗漏判，
                # 交给画回确认门禁兜底，比放宽判据让每题都多几颗幻影划算。
                # 双条件都是为了挡幻影（2026-09-09 调参，样本仅一本题册，换册需重标）：
                # - ring>0.25：真白子 0.25~0.50；手写/阴影造成的假环 <0.2。
                # - core>=med-15：真白子 core-med +2~+27；幻影 core-med 为负。
                if ring > 0.25 and core >= med - 15:
                    white.add((ix, iy))
                    marks.append((cx, cy, r, "W"))
            # 否则：线可见且无描边 → 空点
    return black, white, marks


def classify_stones(gray, xs, ys, circles):
    """圆心吸附到最近交叉点，圆心亮度 <110 判黑。"""
    dx = np.median(np.diff(xs)) if len(xs) > 1 else 20
    dy = np.median(np.diff(ys)) if len(ys) > 1 else 20
    black, white, marks = set(), set(), []
    for cx, cy in circles:
        ix = int(np.argmin([abs(cx - x) for x in xs]))
        iy = int(np.argmin([abs(cy - y) for y in ys]))
        if abs(cx - xs[ix]) > dx * 0.45 or abs(cy - ys[iy]) > dy * 0.45:
            continue
        m = int(gray[max(int(cy) - 3, 0):int(cy) + 4, max(int(cx) - 3, 0):int(cx) + 4].mean())
        (black if m < 110 else white).add((ix, iy))
        marks.append((int(cx), int(cy), int(min(dx, dy) * 0.42), "B" if m < 110 else "W"))
    return black, white, marks


def _blue_mask(bgr):
    """蓝笔手写痕迹的 0/255 mask。蓝墨在灰度图里是暗色（~80-100），
    不洗掉会被扫描分类当成黑子（实测数字"1"让空点 dark_frac=0.53 超标）。

    双阈值：浓蓝（S>=60）之外兼容淡蓝/褪色蓝（H 蓝域 + S>=15 + V>=80）——
    实测浅蓝圆珠笔 S 仅 15-50：S>=60 时 5 个答案数字只检出 1 个，S>=20 仍漏
    最淡的"1"；V>=80 大体挡住黑色印刷线（V 50-90，实测两题图无线段误检，
    残留风险由数字双门槛与画回确认门禁兜底）。铅笔（H~16）不在蓝域，免疫。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 蓝笔：H 90~130（OpenCV 0-180）
    strong = cv2.inRange(hsv, (85, 60, 40), (135, 255, 255))
    faint = cv2.inRange(hsv, (85, 15, 80), (135, 255, 255))
    mask = cv2.bitwise_or(strong, faint)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _blue_mask_strong(bgr):
    """浓蓝单阈值 mask（旧行为）。网格检测阶段专用：淡蓝墨迹在灰度里
    与线同色，留着恰好"桥接"被手写字压断的线；洗掉了线反而断行
    （实测右缘淡蓝数字洗掉后网格丢两行、相位崩）。分类阶段才用双阈值。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (85, 60, 40), (135, 255, 255))
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def detect_blue_digits(bgr, xs, ys, max_seq=9):
    """蓝笔手写编号：蓝色阈值 -> 连通域 -> 吸附交叉点 -> 模板匹配识数。
    返回 [(seq 或 None, ix, iy, conf)]，conf<0.5 的 seq=None（标 ? 请人工确认）。"""
    mask = _blue_mask(bgr)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    dx = np.median(np.diff(xs)) if len(xs) > 1 else 20
    dy = np.median(np.diff(ys)) if len(ys) > 1 else 20
    blobs = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 12 or w > dx * 1.2 or h > dy * 1.2:
            continue
        blobs.append((x, y, w, h, cent[i]))
    # 距离很近的碎片合并（同一数字的两笔）
    merged = []
    used = [False] * len(blobs)
    for i, (x, y, w, h, c) in enumerate(blobs):
        if used[i]:
            continue
        x2, y2, pts = x + w, y + h, [(x, y, w, h)]
        used[i] = True
        for j in range(i + 1, len(blobs)):
            if used[j]:
                continue
            xj, yj, wj, hj, cj = blobs[j]
            if abs(cj[0] - c[0]) < dx * 0.55 and abs(cj[1] - c[1]) < dy * 0.55:
                pts.append((xj, yj, wj, hj))
                used[j] = True
        xa = min(p[0] for p in pts); ya = min(p[1] for p in pts)
        xb = max(p[0] + p[2] for p in pts); yb = max(p[1] + p[3] for p in pts)
        merged.append((xa, ya, xb - xa, yb - ya))
    out = []
    for x, y, w, h in merged:
        cx, cy = x + w / 2, y + h / 2
        ix = int(np.argmin([abs(cx - xx) for xx in xs]))
        iy = int(np.argmin([abs(cy - yy) for yy in ys]))
        seq, conf = _classify_digit(mask[y:y + h, x:x + w])
        out.append((seq if conf >= 0.5 else None, ix, iy, round(conf, 2)))
    # 同一点多个 blob 只留最大 conf
    best = {}
    for s, ix, iy, c in out:
        if (ix, iy) not in best or c > best[(ix, iy)][3]:
            best[(ix, iy)] = (s, ix, iy, c)
    return sorted(best.values(), key=lambda t: (t[0] is None, t[0] or 99))


_DIGIT_TEMPLATES = None


def _detect_raw_lines(gray):
    """HoughLinesP 检所有线段，返回带方向的原始线段列表。
    [(x1,y1,x2,y2,'h'|'v')] —— 供四角精化做直线拟合用。
    与 _detect_lines 的区别：那个只返回位置中值，这个保留端点。"""
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                            minLineLength=w // 8, maxLineGap=10)
    out = []
    if lines is not None:
        for l in lines.reshape(-1, 4):  # cv2 5.x 返回 (N,4)，旧版 (N,1,4)
            x1, y1, x2, y2 = map(float, l)
            dx_, dy_ = abs(x2 - x1), abs(y2 - y1)
            if dy_ < 0.15 * dx_ and dx_ > w // 10:
                # 贴图边的全线是页边/照片边界的阴影伪影，不是棋盘线
                # （真棋盘线贴边 = 拍裁掉了半条线，本身就不该入镜）
                if min(y1, y2) > 6 and max(y1, y2) < h - 6:
                    out.append((x1, y1, x2, y2, 'h'))
            elif dx_ < 0.15 * dy_ and dy_ > h // 10:
                if min(x1, x2) > 6 and max(x1, x2) < w - 6:
                    out.append((x1, y1, x2, y2, 'v'))
    return out


def _fit_line(segs, orient):
    """一簇线段的所有端点最小二乘拟合一条直线。
    横线拟合 y = a x + b；竖线拟合 x = c y + d（防斜率爆炸）。
    返回 (a, b) 使得 orient=='h' 时 y=a x+b，'v' 时 x=a y+b。"""
    xs, ys = [], []
    for x1, y1, x2, y2, o in segs:
        xs += [x1, x2]
        ys += [y1, y2]
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 4:
        return None
    if orient == 'h':
        a, b = np.polyfit(xs, ys, 1)
    else:
        a, b = np.polyfit(ys, xs, 1)
    return float(a), float(b)


def _fit_line_dark(gray, pos, orient, lo, hi, s):
    """弱线的暗像素拟合（_fit_line 的回退）。
    HoughLinesP 检不出的线（被棋子压成碎段达不到 minLineLength），
    逐行/列取带内暗像素平均位置，再拟合直线。
    棋子压线处暗像素虽是整颗子，但子是对称圆、均值仍落在线上，不会带歪。
    返回 (a, b, 覆盖率)；覆盖率 <0.3 说明带内根本没有线，返回 None。"""
    h, w = gray.shape
    band = max(int(s * 0.2), 4)
    dark = gray < 150
    lo, hi = max(int(lo), 0), min(int(hi), gray.shape[0 if orient == 'v' else 1])
    pts_a, pts_b = [], []
    n_rows = 0
    if orient == 'v':
        x0 = max(int(pos - band), 0)
        x1 = min(int(pos + band), w)
        for y in range(lo, hi):
            idx = np.where(dark[y, x0:x1])[0]
            if len(idx):
                pts_a.append(float(idx.mean()) + x0)
                pts_b.append(float(y))
                n_rows += 1
        total = hi - lo
    else:
        y0 = max(int(pos - band), 0)
        y1 = min(int(pos + band), h)
        for x in range(lo, hi):
            idx = np.where(dark[y0:y1, x])[0]
            if len(idx):
                pts_a.append(float(x))
                pts_b.append(float(idx.mean()) + y0)
                n_rows += 1
        total = hi - lo
    if total <= 0 or n_rows / total < 0.3 or len(pts_a) < 4:
        return None
    a, b = np.polyfit(np.array(pts_b), np.array(pts_a), 1)
    return (float(a), float(b), n_rows / total)


def _refine_corners(gray, xs, ys):
    """四角精化：最外 4 条印刷线各自拟合真直线，两两求交得四角。

    为什么必须做（GitHub 五个围棋识别项目的共识）：透视变换保直线，
    所以每条印刷线在照片里仍是一条直线——4 条最外线的交点就是精确四角。
    而均匀网格假设在透视下系统性失真（实测角点拟合偏差 6.5-9px），
    直接拿 xs/ys 外推四角会把 warp 也带歪。失败返回 None（走回退路径）。"""
    s = float(np.median(np.diff(xs))) if len(xs) > 1 else 30
    raw = _detect_raw_lines(gray)
    if not raw:
        return None

    def cluster_near(target, orient):
        tol = s * 0.6
        segs = []
        for x1, y1, x2, y2, o in raw:
            if o != orient:
                continue
            pos = (y1 + y2) / 2 if orient == 'h' else (x1 + x2) / 2
            if abs(pos - target) < tol:
                segs.append((x1, y1, x2, y2, o))
        return segs

    def edge_line(target, orient, lo, hi):
        """先 HoughLinesP 段拟合，段不足回退暗像素拟合（弱线被压碎段）。"""
        segs = cluster_near(target, orient)
        fit = _fit_line(segs, orient) if len(segs) >= 2 else None
        if fit is None:
            dark_fit = _fit_line_dark(gray, target, orient, lo, hi, s)
            fit = (dark_fit[0], dark_fit[1]) if dark_fit else None
        return fit

    top = edge_line(ys[0], 'h', xs[0], xs[-1])
    bot = edge_line(ys[-1], 'h', xs[0], xs[-1])
    left = edge_line(xs[0], 'v', ys[0], ys[-1])
    right = edge_line(xs[-1], 'v', ys[0], ys[-1])
    if None in (top, bot, left, right):
        return None

    def intersect(hline, vline):
        """y = ah x + bh 与 x = av y + bv 的交点。"""
        ah, bh = hline
        av, bv = vline
        # x = av (ah x + bh) + bv  =>  x (1 - av ah) = av bh + bv
        denom = 1 - av * ah
        if abs(denom) < 1e-6:
            return None
        x = (av * bh + bv) / denom
        y = ah * x + bh
        return (float(x), float(y))

    corners = [intersect(top, left), intersect(top, right),
               intersect(bot, left), intersect(bot, right)]
    if None in corners:
        return None
    tl, tr, bl, br = corners
    # 合理性：对边长度比、边与 xs/ys 的端点距离
    w_top = np.hypot(tr[0] - tl[0], tr[1] - tl[1])
    w_bot = np.hypot(br[0] - bl[0], br[1] - bl[1])
    h_left = np.hypot(bl[0] - tl[0], bl[1] - tl[1])
    h_right = np.hypot(br[0] - tr[0], br[1] - tr[1])
    if min(w_top, w_bot) < 0.5 * max(w_top, w_bot):
        return None
    if min(h_left, h_right) < 0.5 * max(h_left, h_right):
        return None
    expect_w = (len(xs) - 1) * s
    expect_h = (len(ys) - 1) * s
    if not (0.6 * expect_w < (w_top + w_bot) / 2 < 1.4 * expect_w):
        return None
    if not (0.6 * expect_h < (h_left + h_right) / 2 < 1.4 * expect_h):
        return None
    return np.float32([tl, tr, bl, br])


_WARP_CELL = 44      # warp 后每格固定 44px：下游所有像素阈值变常量
_WARP_MARGIN = 36    # warp 后四周留白


def _warp_board(bgr, quad, cols, rows):
    """逆透视变换把棋盘拉正成等距正方形网格（最佳实践核心步骤）。
    返回 (warp_bgr, xs, ys)：网格位置变成 margin + i*cell 的精确等距。"""
    w = (cols - 1) * _WARP_CELL + 2 * _WARP_MARGIN
    h = (rows - 1) * _WARP_CELL + 2 * _WARP_MARGIN
    m = float(_WARP_MARGIN)
    dst = np.float32([[m, m], [w - m, m], [m, h - m], [w - m, h - m]])
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(bgr, M, (w, h),
                                 borderValue=(245, 242, 235))
    xs = [m + i * _WARP_CELL for i in range(cols)]
    ys = [m + j * _WARP_CELL for j in range(rows)]
    return warped, xs, ys


def _extend_grid_edges(gray, xs, ys):
    """边缘无子列/行的补救：最外线到图边还有 >0.7s 空间时，
    在外推位置验证是否有弱线证据（被棋子压断的线覆盖率仍应 >35%），
    有则补一列/行，每侧最多补 2 条。

    透视下最外竖线常被压缩导致 HoughLinesP 检不出（实测 6% 透视丢最左列），
    而该列无子就没有圆心证据，点阵拟合的证据范围裁剪（±0.3s）会把它裁掉，
    整盘坐标系统性错位一列。此处用暗像素行/列覆盖率做弱证据验证。"""
    h, w = gray.shape
    if len(xs) < 2 or len(ys) < 2:
        return xs, ys
    s = float(np.median(np.diff(xs)))
    xs, ys = list(xs), list(ys)
    dark = gray < 150

    def line_evidence(pos, orient, lo, hi):
        band = max(int(s * 0.18), 3)
        if orient == 'v':
            x0, x1 = max(int(pos - band), 0), min(int(pos + band), w)
            if x1 - x0 < 2:
                return 0.0
            strip = dark[max(int(lo), 0):min(int(hi), h), x0:x1]
            return float(np.mean(strip.any(axis=1))) if strip.size else 0.0
        y0, y1 = max(int(pos - band), 0), min(int(pos + band), h)
        if y1 - y0 < 2:
            return 0.0
        strip = dark[y0:y1, max(int(lo), 0):min(int(hi), w)]
        return float(np.mean(strip.any(axis=0))) if strip.size else 0.0

    for _ in range(2):
        grew = False
        if xs[0] - s > 10 and line_evidence(xs[0] - s, 'v', ys[0], ys[-1]) > 0.35:
            xs.insert(0, xs[0] - s); grew = True
        if xs[-1] + s < w - 10 and line_evidence(xs[-1] + s, 'v', ys[0], ys[-1]) > 0.35:
            xs.append(xs[-1] + s); grew = True
        if ys[0] - s > 10 and line_evidence(ys[0] - s, 'h', xs[0], xs[-1]) > 0.35:
            ys.insert(0, ys[0] - s); grew = True
        if ys[-1] + s < h - 10 and line_evidence(ys[-1] + s, 'h', xs[0], xs[-1]) > 0.35:
            ys.append(ys[-1] + s); grew = True
        if not grew:
            break
    return xs, ys


def _quad_matches_grid(quad, xs, ys):
    """四角四边形与网格范围一致性校验：边长须在网格跨度的 0.55~1.5 倍内。

    透视会让对边长度不同（梯形畸变），但单边不可能偏离网格跨度一半以上。
    _refine_corners 的暗像素拟合在弱线/棋子压线时会锁到石子边缘，
    给出畸形四边形（实测底边拟合到棋子行，右侧边长仅为网格高的 0.49），
    warp 后整盘压扁。此处拦截，回退原图均匀网格路径。"""
    gw = float(xs[-1] - xs[0])
    gh = float(ys[-1] - ys[0])
    if gw <= 0 or gh <= 0:
        return False
    tl, tr, bl, br = quad

    def dist(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    top, bot = dist(tl, tr), dist(bl, br)
    left, right = dist(tl, bl), dist(tr, br)
    for side, span in ((top, gw), (bot, gw), (left, gh), (right, gh)):
        if not (0.55 * span <= side <= 1.5 * span):
            return False
    return True


def _merge_digits(*digit_lists):
    """多源数字识别结果按交叉点取优。
    规则（两轮实测标定）：**纯按 conf 比较，平票才有读数者优先**——
    "?" 的 conf 是双门槛拦截前的原始分，可能比正确答案还高（透视下
    2→1 的 conf 高达 0.70 但 top2 并列），此时宁可 "?"；反过来 warp 源
    插值模糊产生的低分误读（实测 "2"→"9" conf 0.53）也不该压过原图源
    拦截前 0.62 的判据。误读比 "?" 严重得多：? 走人工确认，错读会
    直接进批改。"""
    best = {}
    for lst in digit_lists:
        for d in lst:
            key = (d[1], d[2])
            cur = best.get(key)
            if cur is None:
                best[key] = d
            elif d[3] > cur[3]:
                best[key] = d
            elif d[3] == cur[3] and cur[0] is None and d[0] is not None:
                best[key] = d
    return sorted(best.values(), key=lambda t: (t[0] is None, t[0] or 99))


def _digit_templates():
    """cv2 渲染 1-9 模板（多字体多粗细），懒加载。
    手写体与印刷体差异大，识别率低是预期——低置信一律标 ? 走人工确认。
    "1" 额外配纯竖杠模板：孩子手写的 1 常是一竖，衬线体 1 反而匹配不上；
    蓝字画在黑子上时模糊晕圈会把字撑胖（晕圈混黑底仍是蓝），竖杠+开运算
    瘦身双管齐下。"""
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        _DIGIT_TEMPLATES = {}
        for d in range(1, 10):
            variants = []
            for font, scale, thick in ((cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2),
                                       (cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2),
                                       (cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3),
                                       (cv2.FONT_HERSHEY_TRIPLEX, 1.0, 2)):
                img = np.zeros((36, 36), np.uint8)
                cv2.putText(img, str(d), (6, 28), font, scale, 255, thick)
                variants.append(img)
            if d == 1:
                bar = np.zeros((36, 36), np.uint8)
                bar[6:30, 15:21] = 255
                variants.append(bar)
            _DIGIT_TEMPLATES[d] = variants
    return _DIGIT_TEMPLATES


def _classify_digit(patch):
    """模板匹配识数。返回 (digit, conf)。patch 为 0/255 mask。
    先补边成方形再缩放，避免长宽比失真。
    双重门槛：conf>=0.5 且 top1-top2 间距>=0.02——透视剪切下
    2→1 的误读 conf 高达 0.70，但 top2 几乎并列（实测间距 0.001），
    宁标 "?" 走人工，不冤枉孩子。"""
    if patch.size == 0 or patch.sum() == 0:
        return None, 0.0
    # 注意：不要在这里做形态学开运算瘦身。蓝字画在黑子上确实会被晕圈撑胖，
    # 但开运算会把白底上的细笔画（干净的 2、7）一起削坏（实测 "2" 的
    # conf 从 0.64 掉到 0.44）。胖块的歧义交给双门槛标 "?" 更安全。
    h, w = patch.shape
    side = max(h, w) + 8
    sq = np.zeros((side, side), np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    sq[y0:y0 + h, x0:x0 + w] = patch
    p = cv2.resize(sq, (36, 36), interpolation=cv2.INTER_AREA)
    scores = []
    for d, variants in _digit_templates().items():
        s = max(cv2.matchTemplate(p, t, cv2.TM_CCOEFF_NORMED)[0][0] for t in variants)
        scores.append((s, d))
    scores.sort(reverse=True)
    best_s, best_d = scores[0]
    margin = best_s - scores[1][0] if len(scores) > 1 else 1.0
    if best_s < 0.5 or margin < 0.02:
        return None, max(0.0, float(best_s))
    return best_d, max(0.0, float(best_s))


def _deskew_image(bgr):
    """拍书页常带透视倾斜（实测竖线偏 6° 而横线水平——侧面拍的梯形畸变），
    先按横竖线主角度做仿射矫正，把网格拉回轴对齐。

    为什么必须做：倾斜下同一条竖线在不同高度的 x 差可达 50px+，
    线段中点聚类会把一条线拆成多个簇、相邻线并入一簇，间距估计
    （实测 63 被估成 43.9）和粗网格全面崩溃，warp 前的四角精化也跟着
     latch 到错误线。矫正后各线残余角度差 <1°（约 ±9px/530px），
    在粗网格容差内；四角精化 + warp 再精确处理剩余透视。
    线段不足或倾角 <0.7° 时原样返回。"""
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray[_blue_mask(bgr) > 0] = 255
    segs = _detect_raw_lines(gray)
    min_len = min(h, w) * 0.25  # 只用长段估角度，短段（棋子弧切线）噪声大
    ah, av = [], []
    for x1, y1, x2, y2, o in segs:
        if (x2 - x1) ** 2 + (y2 - y1) ** 2 < min_len ** 2:
            continue
        a = (math.atan2(y2 - y1, x2 - x1) if o == "h"
             else math.atan2(x2 - x1, y2 - y1))
        # 同一条线的 Hough 端点方向任意（可能反向 180°），不规约的话
        # 中位数会落在 177° 上，仿射直接变成上下翻转（实测整盘棋子镜像）
        a = math.degrees(a) % 180
        if a > 90:
            a -= 180
        if o == "h":
            ah.append(math.radians(a))
        else:
            av.append(math.radians(a))
    if len(ah) < 2 or len(av) < 2:
        return bgr, False
    th, tv = float(np.median(ah)), float(np.median(av))
    if abs(th) < math.radians(0.7) and abs(tv) < math.radians(0.7):
        return bgr, False
    # 仿射：横线方向 u→(1,0)，竖线方向 v→(0,1)
    ux, uy = math.cos(th), math.sin(th)
    vx, vy = math.sin(tv), math.cos(tv)
    T = np.linalg.inv(np.array([[ux, vx], [uy, vy]]))
    corners = np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)
    tc = corners @ T.T
    minxy, maxxy = tc.min(axis=0), tc.max(axis=0)
    M = np.hstack([T, (-minxy).reshape(2, 1)]).astype(np.float32)
    nw = int(math.ceil(maxxy[0] - minxy[0]))
    nh = int(math.ceil(maxxy[1] - minxy[1]))
    out = cv2.warpAffine(bgr, M, (nw, nh), borderValue=(245, 242, 235))
    return out, True


def _peaks(hist, thr, min_gap=6):
    """从 1D 直方图里取超阈值连续段的重心，间隔太近的合并。返回位置列表。"""
    out, i, n = [], 0, len(hist)
    while i < n:
        if hist[i] >= thr:
            j = i
            while j + 1 < n and hist[j + 1] >= thr:
                j += 1
            seg = hist[i:j + 1]
            c = float(np.average(np.arange(i, j + 1), weights=seg))
            if out and c - out[-1] < min_gap:
                out[-1] = (out[-1] + c) * 0.5
            else:
                out.append(c)
            i = j + 1
        else:
            i += 1
    return out


def _vote_hists(gray, max_run=16, thr=195):
    """细游程投票直方图：返回 (colh, rowh)。

    colh[x] = 第 x 列上「细暗游程」覆盖的行数（竖线证据），rowh 同理。
    只找线用（基于光照归一化图），绝不用于棋子分类。
    """
    H, W = gray.shape
    bg = cv2.medianBlur(gray, 91)
    norm = cv2.divide(gray, bg, scale=255)
    dark = norm < thr
    colh = np.zeros(W, float)
    rowh = np.zeros(H, float)
    for y in range(H):
        row = dark[y]
        i = 0
        while i < W:
            if row[i]:
                j = i
                while j + 1 < W and row[j + 1]:
                    j += 1
                if j - i + 1 <= max_run:
                    colh[i:j + 1] += 1
                i = j + 1
            else:
                i += 1
    for x in range(W):
        col = dark[:, x]
        i = 0
        while i < H:
            if col[i]:
                j = i
                while j + 1 < H and col[j + 1]:
                    j += 1
                if j - i + 1 <= max_run:
                    rowh[i:j + 1] += 1
                i = j + 1
            else:
                i += 1
    return colh, rowh


def _refine_lines(gray, vals, s, lo, hi):
    """逐线局部精修：等距拟合只是骨架，真实网格有透视残差（实测行距 106~134px 波动），
    强制等距会在远端漂移半个格距（实测 q3 漂 47px，棋子分类采样点整行落错位置）。

    评分 = 每个候选位置上「暗像素占比」，三道防线缺一不可：
    - 阈值自适应（纸面中位 x0.85）：这两本题册纸面只有 135~150 亮，固定 185 会把
      纸面全判暗，argmax 抓到窗口边缘、整盘网格系统性漂移半个格距（实测）。
    - 并上光照归一化路（norm<195，与投票法同源）：照度不均时单阈值会漏线。
    - 黑子是 86px 的大暗块，对窗口内所有候选位置贡献相同常数，不影响 argmax；
      而 medianBlur 边界效应（黑子密集行 norm 失效）由 gray 阈值路兜住。
    窗口 ±0.45s：< 0.5s 保证不会吸到邻线（邻线至少距 0.55s）。
    """
    H, W = gray.shape
    med = float(np.median(gray))
    bg = cv2.medianBlur(gray, 91)
    norm = cv2.divide(gray, bg, scale=255)
    dark = (gray < med * 0.85) | (norm < 195)
    out = []
    for v in vals:
        a, b = int(round(v - 0.45 * s)), int(round(v + 0.45 * s))
        a, b = max(a, 0), min(b, (H - 1))
        seg = dark[a:b + 1, max(int(lo), 0):min(int(hi), W):4]
        if seg.size == 0:
            out.append(float(v))
            continue
        scores = seg.mean(axis=1)
        out.append(float(a + int(np.argmax(scores))))
    # 安全网：吸附后必须严格单调且间距合理，否则回退骨架值
    ok = all(0.5 * s < out[i + 1] - out[i] < 1.5 * s for i in range(len(out) - 1))
    return out if ok else list(vals)


def detect_grid_vote(gray, max_run=16, thr=195, frac=0.15):
    """细游程投票法检格——detect_grid 出幻影线时的稳健替代。

    为什么需要第二条路：题图照片整体偏暗且带标题文字区时，detect_grid 会把文字行、
    图外空白判成网格线（实测 q3/q4 检成 9 列 x 11 行，凭空多出两行两列幻影线，
    连带在最外一圈认出一簇不存在的棋子）。全局阈值同样失效：纸面整片被判成暗。

    做法：先用 medianBlur 做光照归一化（**只用于找线，绝不用于棋子分类**——核比棋子
    小时大黑子会被当背景除掉），然后逐行找宽度 <= max_run 的细暗游程投给列直方图
    得竖线，逐列同理得横线。棋子是宽游程（直径 ~86px），天然被排除在外。

    返回 (xs, ys, [])（第三项留空以兼容 detect_grid 的返回结构），失败返回 None。
    """
    colh, rowh = _vote_hists(gray, max_run, thr)
    H, W = gray.shape
    xs = _peaks(colh, H * frac)
    ys = _peaks(rowh, W * frac)
    if len(xs) < 4 or len(ys) < 4:
        return None
    return xs, ys, []


def _line_width_at(gray, cx, cy, dx, dy, half=18):
    """过 (cx,cy) 沿法向 (dx,dy) 取 ±half 的一维剖面，返回暗游程宽度（半深法）。

    自适应阈值 = 窗口内纸面分位与最暗值的中点。为什么不用全局阈值：题图亮度差得远，
    固定阈值会把浅色印刷线整个漏掉。为什么不能先做背景减除（gray/medianBlur）：
    模糊核比棋子小时，大黑子会被当成"背景"除掉，黑子在归一化图上反而变成纸面亮度
    （2026-09-09 实测，黑子读数 252），这一步之后所有阈值全部失效。
    返回 None 表示该处没有线（或整窗落在棋子上）。
    """
    H, W = gray.shape
    cx, cy = int(round(cx)), int(round(cy))
    # 采样窗整窗越界直接放弃：幻影线常落在图幅外（实测 q4 最外线 y=1385 = 图高），
    # 不拦会 IndexError；也不能 clamp，clamp 出的假剖面会给出假的线宽。
    if not (half <= cx < W - half and half <= cy < H - half):
        return None
    n = 2 * half + 1
    prof = np.array([gray[int(round(cy + dy * t)), int(round(cx + dx * t))]
                     for t in range(-half, half + 1)], float)
    base = float(np.percentile(prof, 90))
    vmin = float(prof.min())
    if base - vmin < 18:
        return None
    dark = prof < (base + vmin) * 0.5
    c = half
    if not dark[c]:
        return None
    a = c
    while a > 0 and dark[a - 1]:
        a -= 1
    b = c
    while b < n - 1 and dark[b + 1]:
        b += 1
    return int(b - a + 1)


def _grid_quality(gray, xs, ys):
    """给一组线位打分：线上有墨的比例越高越好，间距越均匀越好。

    detect_grid 在带标题文字/手写区的题图上会造出幻影线（实测 q3/q4 检成 9x11），
    幻影线处没有真实墨迹，命中率会把分拉下来；同时它挤在真线之间，间距均匀性变差。
    """
    H, W = gray.shape

    def support(lines, axis, lo, hi):
        rates = []
        for p in lines:
            if not (0 <= p < (W if axis == "v" else H)):
                rates.append(0.0)      # 幻影线可能落在图幅外，直接判零分
                continue
            hit = 0
            ts = np.linspace(lo + 15, hi - 15, 50)
            for t in ts:
                # 采样点同样可能越界（幻影线的 lo/hi 本身就在图外）
                if not (0 <= t < (H if axis == "v" else W)):
                    continue
                w = (_line_width_at(gray, p, t, 1, 0) if axis == "v"
                     else _line_width_at(gray, t, p, 0, 1))
                if w is not None and 2 <= w <= 25:
                    hit += 1
            rates.append(hit / len(ts))
        return float(np.mean(rates)) if rates else 0.0

    sup = 0.5 * (support(xs, "v", ys[0], ys[-1]) + support(ys, "h", xs[0], xs[-1]))
    cvs = []
    for lines in (xs, ys):
        if len(lines) >= 3:
            d = np.diff(lines)
            cvs.append(float(np.std(d) / max(np.mean(d), 1e-6)))
    cv = float(np.mean(cvs)) if cvs else 1.0
    return sup - 0.6 * cv, sup, cv


def _lattice_from_peaks(vals, s_lo=20, s_hi=200):
    """从含杂峰的线位证据里拟合等距点阵（投票法的收尾步骤）。

    投票法给出的峰值里，真网格线的间距会被杂峰切碎——实测 q3 一个 115px 的格子被切成
    40/27/46/44/68 五段。所以不能按近邻合并：那会把两格并成一格（6 个峰缩成 1 条线，
    实际应是 3 条）。正确做法是搜间距 s，用 _fit_lattice 拟合成等距点阵后按"节点有无
    证据支撑"打分：真间距下每个节点附近都有证据；半间距留下一半空节点，倍间距会跳过真线。

    间距初值取"不小于中位数的那些间隔的中位数"（杂峰间隔普遍偏小，会被这一步滤掉），
    再在 ±20~25% 范围内细搜。
    """
    vals = sorted(float(v) for v in vals)
    if len(vals) < 4:
        return None
    d = np.diff(vals)
    big = d[d >= np.median(d)]
    s0 = float(np.median(big)) if len(big) else 0.0
    lo, hi = ((max(s_lo, s0 * 0.80), min(s_hi, s0 * 1.25))
              if 20 <= s0 <= 200 else (s_lo, s_hi))
    best, best_score = None, -1e9
    s = lo
    while s <= hi:
        g = _fit_lattice(vals, s)
        if g and 4 <= len(g) <= 21:
            tol = s * 0.25
            used = sum(1 for n in g if any(abs(v - n) < tol for v in vals))
            err = float(np.mean([min(abs(v - n) for n in g) for v in vals]))
            score = used - 1.2 * (len(g) - used) - err / tol
            if score > best_score:
                best, best_score = g, score
        s += 0.5
    return best


def _flatten_outliers(vals, s):
    """透视矫正后网格线近似等差排布（x = a + b·i）。逐线吸附仍可能被局部
    暗结构带偏（实测 q4 三条竖线被铅笔手写拉偏 16~25px——蓝墨掩码洗不掉
    灰黑铅笔字），用鲁棒直线拟合修正：残差 <=0.15s 的内点保留吸附值
    （保留真实非线性），离群点取拟合值。"""
    idx = np.arange(len(vals), dtype=float)
    v = np.array(vals, float)
    for _ in range(2):
        # Theil-Sen：点对斜率的中位数。最小二乘会被强离群点带歪斜率、
        # 把整组线拉漂（实测 ys[8] 偏 27px 时全盘漂移 27px）；Theil-Sen
        # 容忍 <29% 离群。n<=21，O(n^2) 点对成本可忽略。
        slopes = [(v[j] - v[i]) / (idx[j] - idx[i])
                  for i in range(len(v)) for j in range(i + 1, len(v))]
        b = float(np.median(slopes))
        a = float(np.median(v - b * idx))
        resid = v - (a + b * idx)
        out = np.abs(resid) > 0.15 * s
        if not out.any():
            break
        v[out] = a + b * idx[out]
    return list(v)


def prep_grid(photo_path):
    """识别前段：读图 → 去斜 → 建三张灰图 → 检格 → 延伸 → 吸附。

    抽出来是为了给角定位（corner.py）复用同一套线位：角判定要量「最外四条印刷线」
    的线宽和出头，线位必须和棋子分类用的是同一组，否则两套坐标会各说各话。
    corner 必须用**这一步的 xs/ys（透视矫正后、逆透视 warp 之前）**：
    warp 之后最外四条线被拉到图像边界上，线宽和出头都量不到了。

    返回 dict：bgr / gray / gray_grid / gray_cls / xs / ys / centers
               / orig_bgr / orig_xs / orig_ys（warp 前留档，数字识别用）
    """
    bgr = cv2.imread(photo_path)
    if bgr is None:
        raise ValueError("照片读取失败")
    # 大圧缩到 1600 宽内，稳定检测参数
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    # 透视倾斜预矫正（侧面拍书页竖线会偏 5°+，不矫正粗网格必崩）
    bgr, deskewed = _deskew_image(bgr)
    if deskewed:
        # warpAffine 补边与原图的色差边界会被 Canny 检成斜向假线
        # （实测左缘假线间距 15-20px，直接带崩网格），inpaint 抹平。
        # mask = 与图边连通的精确补边区域，外扩 5px 盖住过渡带。
        # 必须限制连通域：纸面亮部和白子内部可能碰巧同色，
        # 全局匹配会把真实棋盘线/白子也抹掉（实测抹断过两条横线）。
        fill = np.all(bgr == (245, 242, 235), axis=2).astype(np.uint8)
        n, labels = cv2.connectedComponents(fill)
        edge_ids = set(np.unique(np.concatenate(
            [labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        if edge_ids:
            mask = np.isin(labels, list(edge_ids)).astype(np.uint8) * 255
            mask = cv2.dilate(mask, np.ones((11, 11), np.uint8))
            bgr = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 蓝墨在灰度里是暗色，会污染黑子判定（圆盘暗像素占比）和检圆亮度验证。
    # 网格检测/吸附阶段用强蓝洗（gray_grid）：淡蓝墨迹留着能桥接被压断的线；
    # 棋子分类阶段用双阈值洗（gray_cls）：淡蓝也是暗色污染，必须除尽。
    gray_grid = gray.copy()
    gray_grid[_blue_mask_strong(bgr) > 0] = 255
    gray_cls = gray.copy()
    gray_cls[_blue_mask(bgr) > 0] = 255

    out = detect_grid(gray_grid)
    if out is None:
        raise ValueError("未检测到棋盘网格，请重拍（正对题图、光线均匀、题图完整入镜）")
    xs, ys, centers = out
    xs, ys = _extend_grid_edges(gray_grid, xs, ys)
    # 网格线吸附：圆心锚定的点阵有系统性偏移（边石子中心偏出+s累积），
    # >10px 时扫描分类采样带套不住真线，邻子描边环进带致白子判空
    xs, ys = _snap_grid_to_lines(gray_grid, xs, ys, centers)

    # 幻影线兜底：detect_grid 在带标题文字/手写区的题图上会把图外内容判成网格线
    # （实测 q3/q4 检成 9x11，凭空多两行两列，最外一圈还会认出一簇不存在的棋子）。
    # 只在两套方法给出**不同行列数**时才允许切换，且必须质量分明显更高——
    # 行列数一致时一律沿用原结果，避免动到已人工核对过的题图。
    vg = detect_grid_vote(gray_grid)
    if vg:
        vx, vy = _lattice_from_peaks(vg[0]), _lattice_from_peaks(vg[1])
        if vx and vy and (len(vx) != len(xs) or len(vy) != len(ys)):
            q_old = _grid_quality(gray_grid, xs, ys)[0]
            q_new = _grid_quality(gray_grid, vx, vy)[0]
            if q_new > q_old + 0.05:
                # 等距拟合只是骨架：真实网格有透视残差（行距 106~134px 波动），
                # 必须逐线吸附回真实峰位，否则远端漂移半个格距
                s_v = float(np.median(np.diff(vx)))
                s_h = float(np.median(np.diff(vy)))
                xs = _flatten_outliers(
                    _refine_lines(gray_grid, vx, s_v, vy[0] - s_h, vy[-1] + s_h), s_v)
                ys = _flatten_outliers(
                    _refine_lines(gray_grid, vy, s_h, vx[0] - s_v, vx[-1] + s_v), s_h)
    return {
        "bgr": bgr, "gray": gray, "gray_grid": gray_grid, "gray_cls": gray_cls,
        "xs": xs, "ys": ys, "centers": centers,
        "orig_bgr": bgr, "orig_xs": list(xs), "orig_ys": list(ys),
    }


def recognize(photo_path, read_digits=False):
    """主入口。返回 dict：网格线数、黑白子（网格坐标）、蓝字编号、叠加核对图路径。

    read_digits：手写数字（作答顺序）识别开关，**2026-09-08 起默认关闭**。
    用户判定该能力无实际价值（铅笔字读不出、圆珠笔读数要靠人复核，
    省不掉人工却多一个误判来源），功能取消。代码保留以便日后回滚。
    关闭时 digits 恒为空，识别提速且不产生任何"?"读数噪声。

    V2 管线（对标 GitHub 围棋识别最佳实践 GoChessParse/image2sgf/GOimage2SGF）：
    1. 粗网格检测（detect_grid）拿行列数和粗网格
    2. 四角精化（最外 4 条印刷线拟合求交）→ 逆透视变换拉正
    3. warp 后网格严格等距（cell=44 常量），全交叉点扫描分类
    4. 蓝字编号也在 warp 后图上识别（透视被矫正，模板匹配更准）
    四角精化失败回退旧路径（原图均匀网格），保证不回归。"""
    P = prep_grid(photo_path)
    bgr, gray_grid, gray_cls = P["bgr"], P["gray_grid"], P["gray_cls"]
    xs, ys = P["xs"], P["ys"]
    # 原图留档：数字识别融合用（正视角下原图无插值模糊，数字更准）
    orig_bgr, orig_xs, orig_ys = P["orig_bgr"], P["orig_xs"], P["orig_ys"]

    quad = _refine_corners(gray_grid, xs, ys)
    if quad is not None and not _quad_matches_grid(quad, xs, ys):
        # 角点拟合失败会给出畸形四边形（实测底线拟合到棋子上，
        # 右侧边长只有网格高的一半），warp 后整盘压扁分类全崩
        quad = None
    if quad is not None:
        # V2 主路径：逆透视拉正，网格变成精确等距常量
        wb, xs, ys = _warp_board(bgr, quad, len(xs), len(ys))
        gray_cls = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
        gray_cls[_blue_mask(wb) > 0] = 255
        bgr = wb

    black, white, marks = classify_by_sweep(gray_cls, xs, ys,
                                            ignore=_ignore_mask(bgr))
    if read_digits:
        # 手写数字识别（已默认关闭，见 recognize 文档）
        if quad is not None:
            digits = _merge_digits(
                detect_blue_digits(orig_bgr, orig_xs, orig_ys),
                detect_blue_digits(bgr, xs, ys))
        else:
            digits = detect_blue_digits(bgr, xs, ys)
    else:
        digits = []

    overlay = bgr.copy()
    for x in xs:
        cv2.line(overlay, (int(x), int(min(ys))), (int(x), int(max(ys))), (0, 255, 0), 1)
    for y in ys:
        cv2.line(overlay, (int(min(xs)), int(y)), (int(max(xs)), int(y)), (0, 255, 0), 1)
    for cx, cy, cr, c in marks:
        cv2.circle(overlay, (cx, cy), cr, (0, 0, 255) if c == "B" else (255, 0, 255), 2)
    for s, ix, iy, c in digits:
        cv2.putText(overlay, str(s or "?"), (int(xs[ix]) - 8, int(ys[iy]) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    overlay_path = photo_path.rsplit(".", 1)[0] + "_grid.png"
    cv2.imwrite(overlay_path, overlay)

    return {
        "cols": len(xs), "rows": len(ys),
        "black": sorted(black), "white": sorted(white),
        "digits": digits,
        "overlay": overlay_path,
    }


def _spacing_mode(vals):
    """从一组线位里估格距：相邻间距互相投票（容差 ±15%），取票数最多的那个。

    为什么不用中位数：多题拼在一张图时，块间的大间距只有一两个，块内格距有十几个，
    中位数在题数接近半数时才被带偏，而投票法在 >50% 的间距都是格距时必然取到格距。
    """
    if len(vals) < 3:
        return None
    ds = np.asarray(np.diff(vals), float)
    ds = ds[(ds >= 8) & (ds <= 400)]
    if len(ds) == 0:
        return None
    best, best_n = None, 0
    for d in ds:
        n = int(np.sum(np.abs(ds - d) <= d * 0.15))
        # 平票时取**更大**的间距：杂峰（石子边缘/文字）只会制造比真格距更小的
        # 间距，取小会踩坑——实测合成多题图上真格距 44 与杂峰间距 12/27 平票，
        # 取小得到 12，按 1.7*12 断块把每行劈成碎片，裁出来只有 72px 高，
        # 连网格都检不出（2026-09-09）。
        if n > best_n or (n == best_n and best is not None and d > best):
            best, best_n = float(d), n
    return best


def _group_lines(vals, s, min_lines=4):
    """按间距断块：相邻线位间隔 > 1.7 个格距就认为是两道题。
    少于 min_lines 条的碎块（文字行、噪线）丢掉。"""
    groups, cur = [], [vals[0]]
    for a, b in zip(vals, vals[1:]):
        if b - a > 1.7 * s:
            groups.append(cur)
            cur = [b]
        else:
            cur.append(b)
    groups.append(cur)
    return [g for g in groups if len(g) >= min_lines]


def split_boards(photo_path, frac=0.10, min_lines=4):
    """一图多题切分（印刷题图专用）。返回裁剪图路径列表，单棋盘时长度为 1。

    架构与 GitHub 多棋盘识别项目（kaya-go/moku、Chess_diagram_to_FEN）的共识一致：
    「先切出每个棋盘区域 → 每块独立走单盘管线」。切分本身用传统 CV 就够
    （深度学习目标检测太重，且 moku 是 AGPL 会传染公开仓库）。

    **2026-09-09 重写**：旧版走 HoughLinesP 线段 + 包围盒并查集聚类，在真实题图上
    实测不可用——拼 2 题/3 题的合成图都只切出 1 块：印刷网格线细且常被棋子压断，
    Hough 只检到 17 条线段（两块 7x9 应有 30+ 条），每块都凑不齐"横竖各≥2 条"的
    簇门槛，只剩一个簇。旧实现保留在 _split_boards_hough 供对照。

    新版做法：复用单盘管线的细游程投票直方图（_vote_hists，本就是为了从弱线里找
    网格而写的，比 Hough 稳得多）→ 取全图所有网格线峰位 → 投票估格距 →
    按"间隔 > 1.7 格距"断成若干横/竖线组 → 横竖组笛卡尔积得到候选题块 →
    每块裁下来重新投票自检（真块里必然还有 ≥4 条横竖线，错配的空白区检不到线，
    用于过滤错位排版的伪组合）→ 阅读顺序排序 → 加边裁剪。

    局限：① 两题间距 < 1.7 格距时会并成一块（此时整块走单盘管线，由画回确认
    门禁兜底）；② 面向真实木盘/复杂背景不适用（同 classify_by_sweep 的说明）。
    """
    bgr = cv2.imread(photo_path)
    if bgr is None:
        raise ValueError("照片读取失败")
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray[_blue_mask(bgr) > 0] = 255  # 手写蓝字会截断线段，先洗掉
    H, W = gray.shape

    colh, rowh = _vote_hists(gray)
    xs = _peaks(colh, H * frac)
    ys = _peaks(rowh, W * frac)
    if len(xs) < min_lines or len(ys) < min_lines:
        return [photo_path]
    sx, sy = _spacing_mode(xs), _spacing_mode(ys)
    if not sx or not sy:
        return [photo_path]
    gx = _group_lines(xs, sx, min_lines)
    gy = _group_lines(ys, sy, min_lines)
    if len(gx) <= 1 and len(gy) <= 1:
        return [photo_path]

    # 加边：0.35 个格距（下限 20px）。太小切出来的块连网格都检不出——
    # 实测 0.12s（s=44 时只有 10px）的合成块跑 prep_grid 直接报"未检测到网格"，
    # 因为最外两条印刷线贴着图边，线宽/出头都没法量（同 corner 的注意事项）。
    pad = int(max(20, 0.35 * min(sx, sy)))
    regions = []
    for vx in gx:
        for hy in gy:
            x1, x2 = int(vx[0]) - pad, int(vx[-1]) + pad
            y1, y2 = int(hy[0]) - pad, int(hy[-1]) + pad
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, W), min(y2, H)
            crop = gray[y1:y2, x1:x2]
            if crop.shape[0] < 4 * sy or crop.shape[1] < 4 * sx:
                continue
            # 自检：真题块里横竖线都还在，错配的空白区检不出线
            c2, r2 = _vote_hists(crop)
            if (len(_peaks(c2, crop.shape[0] * frac)) < min_lines
                    or len(_peaks(r2, crop.shape[1] * frac)) < min_lines):
                continue
            regions.append((x1, y1, x2, y2))

    if len(regions) <= 1:
        return [photo_path]

    # 阅读顺序：先按行带（纵向分组）再按横向位置
    med_h = sorted(r[3] - r[1] for r in regions)[len(regions) // 2]
    band = max(med_h * 0.6, 1)
    regions.sort(key=lambda r: (int((r[1] + r[3]) / 2) // band, r[0]))

    # 写盘必须无损：白子判据（描边环采样，暗阈值 <120）对 JPEG 极敏感——
    # 实测同一张图另存 jpg 后白子漏 7/7（q100 也漏 5/7），存 png 漏 0。
    # 切分裁剪是有损链路上最容易被忽略的一环，这里统一用 png（2026-09-09）。
    stem = photo_path.rsplit(".", 1)[0]
    out = []
    for i, (x1, y1, x2, y2) in enumerate(regions):
        p = f"{stem}_b{i}.png"
        cv2.imwrite(p, bgr[y1:y2, x1:x2])
        out.append(p)
    return out


def _split_boards_hough(photo_path):
    """旧版切分（HoughLinesP 线段 + 并查集聚类），2026-09-09 被 split_boards 取代。

    保留原因：在新版表现不好的场景（例如题块贴得极近、或线极粗的数码图）
    可以切回来做 A/B 对照，评测脚本直接换函数名即可。
    """
    bgr = cv2.imread(photo_path)
    if bgr is None:
        raise ValueError("照片读取失败")
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray[_blue_mask(bgr) > 0] = 255  # 手写蓝字会截断线段，先洗掉

    segs = _detect_raw_lines(gray)
    if len(segs) < 4:
        return [photo_path]

    n = len(segs)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    boxes = [(min(s[0], s[2]), min(s[1], s[3]),
              max(s[0], s[2]), max(s[1], s[3])) for s in segs]
    thr = max(8, min(bgr.shape[:2]) // 100)
    for i in range(n):
        bi = boxes[i]
        for j in range(i + 1, n):
            bj = boxes[j]
            if not (bi[2] + thr < bj[0] or bj[2] + thr < bi[0]
                    or bi[3] + thr < bj[1] or bj[3] + thr < bi[1]):
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters = []
    for idxs in groups.values():
        hs = sum(1 for i in idxs if segs[i][4] == "h")
        vs = sum(1 for i in idxs if segs[i][4] == "v")
        x1 = min(boxes[i][0] for i in idxs)
        y1 = min(boxes[i][1] for i in idxs)
        x2 = max(boxes[i][2] for i in idxs)
        y2 = max(boxes[i][3] for i in idxs)
        if hs >= 2 and vs >= 2 and (x2 - x1) >= 60 and (y2 - y1) >= 60:
            clusters.append((x1, y1, x2, y2))

    merged_any = True
    while merged_any:
        merged_any = False
        for i in range(len(clusters)):
            if merged_any:
                break
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    clusters[i] = (min(a[0], b[0]), min(a[1], b[1]),
                                   max(a[2], b[2]), max(a[3], b[3]))
                    clusters.pop(j)
                    merged_any = True
                    break
    if len(clusters) <= 1:
        return [photo_path]

    med_h = sorted(c[3] - c[1] for c in clusters)[len(clusters) // 2]
    band = max(med_h, 1) * 1.2
    clusters.sort(key=lambda c: (int(c[1] + (c[3] - c[1]) / 2) // band,
                                 c[0] + (c[2] - c[0]) / 2))

    stem = photo_path.rsplit(".", 1)[0]
    ext = photo_path.rsplit(".", 1)[1] if "." in photo_path else "jpg"
    pad = 14
    ih, iw = bgr.shape[:2]
    out = []
    for i, (x1, y1, x2, y2) in enumerate(clusters):
        crop = bgr[max(int(y1) - pad, 0):min(int(y2) + pad, ih),
                   max(int(x1) - pad, 0):min(int(x2) + pad, iw)]
        p = f"{stem}_b{i}.{ext}"
        cv2.imwrite(p, crop)
        out.append(p)
    return out
