"""棋形识别：纸质死活题照片 -> 网格坐标。

经验（来自 SKILL.md 校对清单，勿回退）：
- 不用投影法找棋盘线（棋子主导投影出假线）——用 HoughLinesP 检测印刷线 + 圆心聚类兜底
- 不用交叉点暗像素分类（白子细边与空点线十字不可分）——用 HoughCircles + 圆心亮度
- HoughCircles param2=28 紧凑可靠；param2=25 会把蓝字/印刷字误检为白子
- 产出叠加核对图（_grid.png），人工核对每颗子是必须质检步骤
- 手写答案双人复核：识别出的编号序列必须列给用户确认，读不准标 "?"
"""
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
                line_ys.append((y1 + y2) / 2)
            elif dx_ < 0.15 * dy_ and dy_ > h // 8:
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
    s0 = _spacing_prior(blacks) or _spacing_prior(c1)
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


def _line_visibility(gray, cx, cy, r, edge_l, edge_r, edge_t, edge_b):
    """网格线在该节点是否可见（印刷图"有子必盖线"）。返回 (横线占比, 竖线占比)。

    三个关键设计，全是踩坑换来的：
    - 粗采样带 ±10px 再投影成 1D：节点拟合常有偏差（角点证据少、透视下
      实测达 6.5-9px），窄带采样会整条线漏掉（实测角点 cross_dark=4 vs 中部 34）
    - 边缘/角节点只采棋盘内侧半窗：边节点外侧本来就没有线，
      全窗采样会让暗像素天然减半，固定阈值必然把边缘空点误判成白子
    - 按采样长度归一化成占比而非绝对像素数：间距 s 变化时绝对数不可比"""
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
    hf = float(np.mean((rows < 150).any(axis=0))) if rows.size else 0.0
    y0 = cy if edge_t else max(cy - r, 0)
    y1 = cy if edge_b else min(cy + r, h - 1)
    if y1 <= y0:
        y0, y1 = max(cy - r, 0), min(cy + r, h - 1)
    cols = gray[y0:y1 + 1, max(cx - band, 0):cx + band + 1]
    vf = float(np.mean((cols < 150).any(axis=1))) if cols.size else 0.0
    return hf, vf


def classify_by_sweep(gray, xs, ys):
    """全交叉点扫描分类（印刷题图专用，比 HoughCircles 稳一个量级）：
    - 中心圆盘暗像素占比 >0.5 → 黑子（比单点亮度抗噪：空点中心压线也偏暗，
      实测空点中心亮度 116，距旧阈值 110 仅一线之隔，但圆盘占比仅 ~0.3）
    - 否则横竖任一线可见（占比 >0.4）→ 空点
    - 否则 → 白子（网格线被白子盖住了）
    依据：印刷图里"有子必盖线"。真实木盘照片不适用（木纹/透视），勿挪用。"""
    dx = np.median(np.diff(xs)) if len(xs) > 1 else 20
    dy = np.median(np.diff(ys)) if len(ys) > 1 else 20
    r = int(min(dx, dy) * 0.35)
    nx, ny = len(xs), len(ys)
    black, white, marks = set(), set(), []
    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            cx, cy = int(x), int(y)
            rd = max(r // 2, 3)
            disc = gray[max(cy - rd, 0):cy + rd + 1, max(cx - rd, 0):cx + rd + 1]
            dark_frac = float(np.mean(disc < 110)) if disc.size else 0.0
            if dark_frac > 0.5:
                black.add((ix, iy))
                marks.append((cx, cy, r, "B"))
                continue
            hf, vf = _line_visibility(gray, cx, cy, r,
                                      ix == 0, ix == nx - 1,
                                      iy == 0, iy == ny - 1)
            if max(hf, vf) > 0.4:
                pass  # 空点
            else:
                white.add((ix, iy))
                marks.append((cx, cy, r, "W"))
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


def _digit_templates():
    """cv2 渲染 1-9 模板（多字号多粗细），懒加载。
    手写体与印刷体差异大，识别率低是预期——低置信一律标 ? 走人工确认。"""
    global _DIGIT_TEMPLATES
    if _DIGIT_TEMPLATES is None:
        _DIGIT_TEMPLATES = {}
        for d in range(1, 10):
            variants = []
            for scale, thick in ((0.9, 2), (1.1, 2), (1.0, 3)):
                img = np.zeros((36, 36), np.uint8)
                cv2.putText(img, str(d), (6, 28), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, 255, thick)
                variants.append(img)
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


def recognize(photo_path):
    """主入口。返回 dict：网格线数、黑白子（网格坐标）、蓝字编号、叠加核对图路径。"""
    bgr = cv2.imread(photo_path)
    if bgr is None:
        raise ValueError("照片读取失败")
    # 大圧缩到 1600 宽内，稳定检测参数
    h, w = bgr.shape[:2]
    if max(h, w) > 1600:
        s = 1600 / max(h, w)
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # 蓝墨在灰度里是暗色，会污染黑子判定（圆盘暗像素占比）和检圆亮度验证，
    # 所有棋形识别一律用洗掉蓝字后的灰度图；数字识别仍用原彩图
    gray_cls = gray.copy()
    gray_cls[_blue_mask(bgr) > 0] = 255

    out = detect_grid(gray_cls)
    if out is None:
        raise ValueError("未检测到棋盘网格，请重拍（正对题图、光线均匀、题图完整入镜）")
    xs, ys, centers = out
    black, white, marks = classify_by_sweep(gray_cls, xs, ys)
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
