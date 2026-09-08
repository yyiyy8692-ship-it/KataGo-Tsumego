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
        if k >= 2:
            if k > 4 or abs(gap / k - s) >= 0.3 * s:
                return None
            for i in range(1, k):
                out.append(a + gap * i / k)
        out.append(b)
    return out if len(out) >= 3 else None


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
    if not xs or not ys:
        # 回退路径：线太少时靠圆心证据的点阵/融合拟合（勿动，老路兜底）
        xs = ys = None
        if s0:
            xs = _fit_lattice(line_xs + [c[0] for c in c1], s0)
            ys = _fit_lattice(line_ys + [c[1] for c in c1], s0)
        if not xs or not ys:
            xs = _fuse(line_xs, [c[0] for c in c1])
            ys = _fuse(line_ys, [c[1] for c in c1])
        if not xs or not ys:
            return None
        s = min(np.median(np.diff(xs)) if len(xs) > 1 else 1e9,
                np.median(np.diff(ys)) if len(ys) > 1 else 1e9)
        if s == 1e9 or s < 8:
            return xs, ys, c1
        c2 = _consensus_circles(gray, s)
        if c2:
            # 精检圆心 + 线联合证据再拟合：补被棋子压断的线
            xs2 = _fit_lattice(line_xs + [c[0] for c in c2], s)
            ys2 = _fit_lattice(line_ys + [c[1] for c in c2], s)
            if xs2 and len(xs2) >= len(xs):
                xs = xs2
            if ys2 and len(ys2) >= len(ys):
                ys = ys2
            return xs, ys, c2
    return xs, ys, c1


def grid_from_stones(centers):
    """线检测失败时，用棋子圆心坐标聚类出网格（兜底）。"""
    xs = _cluster([c[0] for c in centers], 12)
    ys = _cluster([c[1] for c in centers], 12)
    return (xs, ys) if len(xs) >= 2 and len(ys) >= 2 else None


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
            if dark_frac > 0.5:
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
                if ring > 0.25 or (ring > 0.14 and min(hf, vf) < 0.95):
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
    不洗掉会被扫描分类当成黑子（实测数字"1"让空点 dark_frac=0.53 超标）。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # 蓝笔：H 90~130（OpenCV 0-180）
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


def recognize(photo_path):
    """主入口。返回 dict：网格线数、黑白子（网格坐标）、蓝字编号、叠加核对图路径。

    V2 管线（对标 GitHub 围棋识别最佳实践 GoChessParse/image2sgf/GOimage2SGF）：
    1. 粗网格检测（detect_grid）拿行列数和粗网格
    2. 四角精化（最外 4 条印刷线拟合求交）→ 逆透视变换拉正
    3. warp 后网格严格等距（cell=44 常量），全交叉点扫描分类
    4. 蓝字编号也在 warp 后图上识别（透视被矫正，模板匹配更准）
    四角精化失败回退旧路径（原图均匀网格），保证不回归。"""
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
    # 蓝墨在灰度里是暗色，会污染黑子判定（圆盘暗像素占比）和检圆亮度验证，
    # 所有棋形识别一律用洗掉蓝字后的灰度图；数字识别仍用原彩图
    gray_cls = gray.copy()
    gray_cls[_blue_mask(bgr) > 0] = 255

    out = detect_grid(gray_cls)
    if out is None:
        raise ValueError("未检测到棋盘网格，请重拍（正对题图、光线均匀、题图完整入镜）")
    xs, ys, centers = out
    xs, ys = _extend_grid_edges(gray_cls, xs, ys)

    # 原图留档：数字识别融合用（正视角下原图无插值模糊，数字更准）
    orig_bgr, orig_xs, orig_ys = bgr, list(xs), list(ys)

    quad = _refine_corners(gray_cls, xs, ys)
    if quad is not None:
        # V2 主路径：逆透视拉正，网格变成精确等距常量
        wb, xs, ys = _warp_board(bgr, quad, len(xs), len(ys))
        gray_cls = cv2.cvtColor(wb, cv2.COLOR_BGR2GRAY)
        gray_cls[_blue_mask(wb) > 0] = 255
        bgr = wb

    black, white, marks = classify_by_sweep(gray_cls, xs, ys,
                                            ignore=_ignore_mask(bgr))
    if quad is not None:
        digits = _merge_digits(detect_blue_digits(orig_bgr, orig_xs, orig_ys),
                               detect_blue_digits(bgr, xs, ys))
    else:
        digits = detect_blue_digits(bgr, xs, ys)

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


def split_boards(photo_path):
    """一图多题切分（印刷题图专用）。返回裁剪图的路径列表，单棋盘时长度为 1。

    GitHub 多棋盘识别项目（kaya-go/moku、tsoj/Chess_diagram_to_FEN 等）的共识
    架构是「先切分出每个棋盘区域 → 每个区域独立走单盘管线」。切分用深度学习
    目标检测太重且 moku 是 AGPL 协议（会传染公开仓库），印刷题图用传统 CV 即可：
    HoughLinesP 检所有线段 → 线段包围盒按空间邻接做并查集聚类——同一棋盘的
    横竖线互相交叉接触必然连通，题与题之间的空白天然断开 → 过滤噪声簇
    （横竖线各不足 2 条、或包围盒太小的丢掉）→ 阅读顺序排序 → 加边裁剪。

    局限：两个棋盘贴得比邻接阈值还近时会并成一簇（此时整簇走单盘管线，
    步骤②画回确认门禁兜底）；面向真实木盘/复杂背景不适用（同 classify_by_sweep）。
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
    # 邻接阈值：图片短边的 1%（下限 8px）。棋盘内部横竖线互相交叉，远小于此；
    # 印刷题册题间距通常 ≥ 2 个格宽，远大于此。
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
        # 噪声过滤：真棋盘横竖线各 ≥2 条且尺寸像块棋盘
        if hs >= 2 and vs >= 2 and (x2 - x1) >= 60 and (y2 - y1) >= 60:
            clusters.append((x1, y1, x2, y2))
    if len(clusters) <= 1:
        return [photo_path]

    # 阅读顺序：先按行带（纵坐标分组）再按横坐标
    med_h = sorted(c[3] - c[1] for c in clusters)[len(clusters) // 2]
    band = max(med_h, 1) * 1.2
    clusters.sort(key=lambda c: (int(c[1] + (c[3] - c[1]) / 2) // band,
                                 c[0] + (c[2] - c[0]) / 2))

    stem = photo_path.rsplit(".", 1)[0]
    ext = photo_path.rsplit(".", 1)[1] if "." in photo_path else "jpg"
    pad = 14  # 加边：给四角精化留点余量
    ih, iw = bgr.shape[:2]
    out = []
    for i, (x1, y1, x2, y2) in enumerate(clusters):
        crop = bgr[max(int(y1) - pad, 0):min(int(y2) + pad, ih),
                   max(int(x1) - pad, 0):min(int(x2) + pad, iw)]
        p = f"{stem}_b{i}.{ext}"
        cv2.imwrite(p, crop)
        out.append(p)
    return out
