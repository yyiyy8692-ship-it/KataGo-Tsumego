"""轻量版讲题：正解 + 3~5 步变化图 + 讲解（不需要录入孩子的下法）。

为什么要单独一套，而不复用 grader.grade()：
    grade() 是给**空枰死活题**设计的，结尾强制走死活裁决
    (_verdict_from_ownership：用 ownership 平均归属判目标棋群死活)。
    「全局题」是 19 路实战局面（60~105 子），「手筋题」目标是吃子/连通/破空，
    两者都没有"某块棋是死是活"这个目标，套上去只会输出无意义的活/死/不明。
    轻量版改为：引擎正解 + 双方最优应对的变化序列 + 目差，不做死活裁决。

关键约定（2026-09-10 实测确认，勿凭直觉改）：
- KataGo 的 scoreLead 是**黑方固定视角**（正数=黑领先），不随走棋方翻转。
  判据：让黑走一手 pass（局面不变）后，scoreLead 从 -9.33 变成 -17.92
  ——pass 亏一手、黑更落后，符合"固定黑视角"；若视角随走棋方翻转，pass 后
  会变成"黑领先 17.92"，荒谬。因此变化图里各手的目差可以直接横向比较，
  第 i 手的收益 = lead[i] - lead[i-1]（黑视角，正=黑得利）。
- 小棋盘（手筋题）komi 传 0：题目不贴目，带 7.5 会把净活的棋块读成劣势。
- **别用 gain（相邻两手目差之差）衡量某一手的价值**：turns[0] 的 scoreLead
  已经包含了"黑先走"的收益，沿着最优 PV 走下去双方都是最优应对，目差几乎不
  动（实测手筋1 四手的 gain 是 +0.1/-0.1/-0.0/-0.0）。某手值多少目要看**该
  局面 moveInfos 里首选与次选的差**（cands[0] - cands[1]，手筋1 是 2.4 目）。
  变化图因此只标各手的绝对目差，不标 gain。
- maxVisits 建议 >=3000：实测手筋3 在 800 visits 时 G6/F3 并列（11.0/10.6），
  8000 才拉开 0.4 目 —— 低深度分不出正解，讲解会指错方向。
"""
import html

from recognition.board import board_parts, C_BG, C_TEXT, C_SUB, C_MARK
from engine.katago import get_engine, best_moves, move_infos, gtp, from_gtp


# 「最强应手」的选点策略（2026-09-11 实测定参，勿凭直觉改）
MIN_VISITS = 30      # 候选参与目差比较的最低访问量：低于此的 scoreLead 是噪声
MIN_GAIN = 0.3       # 目差准则要推翻引擎首选，至少得便宜这么多目


class GoBoard:
    """最小围棋规则：落子 + 提子。够推演引擎 PV 用（PV 里不会出现打劫）。"""

    def __init__(self, cols, rows, black=(), white=()):
        self.cols, self.rows = cols, rows
        self.b = set(map(tuple, black))
        self.w = set(map(tuple, white))

    def _nb(self, x, y):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.cols and 0 <= ny < self.rows:
                yield nx, ny

    def _group(self, x, y, stones):
        """返回 (该棋块所有点, 所有气的点)。"""
        seen, stack, libs = {(x, y)}, [(x, y)], set()
        occupied = self.b | self.w
        while stack:
            cx, cy = stack.pop()
            for nx, ny in self._nb(cx, cy):
                if (nx, ny) in stones:
                    if (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
                elif (nx, ny) not in occupied:
                    libs.add((nx, ny))
        return seen, libs

    def play(self, color, x, y):
        """落子并提掉无气的块，返回本手从盘上移除的点（含被提与自杀）。"""
        mine = self.b if color == "B" else self.w
        theirs = self.w if color == "B" else self.b
        mine.add((x, y))
        removed = set()
        for nx, ny in self._nb(x, y):
            if (nx, ny) in theirs:
                grp, libs = self._group(nx, ny, theirs)
                if not libs:
                    removed |= grp
        theirs -= removed
        grp, libs = self._group(x, y, mine)
        if not libs:                      # 中国规则禁自杀，PV 里不该出现
            mine -= grp
            removed |= grp
        return removed

    def stones(self):
        return sorted(self.b), sorted(self.w)


def _default_komi(cols, rows):
    return 7.5 if min(cols, rows) >= 15 else 0.0


def _pick(moveInfos, to_play, tough=True):
    """从候选里挑一手，返回 (move, lead, criterion, visits)。

    策略（2026-09-11 改，理由见 tests 与 diag_bestmove.py 实测）：
    1. **默认信引擎首选**。KataGo 的 moveInfos 已按它自己的稳健序（LCB）排好，
       不是裸 scoreLead 排的。实测胜率饱和局面里，只被访问 1 次的候选会给出
       离谱的 scoreLead（全局题2 白方 K10：lead=+0.27、vis=1，实算 -9 目），
       无门槛地按 scoreLead 重排必翻车。
    2. **tough（最顽强）**：在访问量 >= MIN_VISITS 的候选里，黑取 scoreLead
       最大、白取最小（= 让黑领先最少，即最顽强的抵抗）。只有比引擎首选便宜
       >= MIN_GAIN 目才真的替换——差个零点几目说明是搜索噪声，不值得换。
    3. 这样"讲题用的最强应手"= 引擎最优，且在大胜/大败已定、引擎开始随手时
       仍能靠目差把它拽回最顽强的一手。
    """
    mis = list(moveInfos)
    first = mis[0]
    fm, fl = first["move"], float(first.get("scoreLead", 0.0))
    if not tough or len(mis) == 1:
        return fm, fl, "引擎首选", int(first.get("visits", 0))
    pool = [m for m in mis if int(m.get("visits", 0)) >= MIN_VISITS] or [first]
    want = min if to_play == "W" else max        # 白要让黑领先最少
    cand = want(pool, key=lambda m: float(m.get("scoreLead", 0.0)))
    cl = float(cand.get("scoreLead", 0.0))
    if cand["move"] == fm or abs(cl - fl) < MIN_GAIN:
        return fm, fl, "引擎首选", int(first.get("visits", 0))
    return cand["move"], cl, "目差最顽强", int(cand.get("visits", 0))


def _root(eng, size, black, white, to_play, visits, komi):
    """根查询，返回 (turn, moveInfos, lead0)。"""
    t = eng.query(size, black, white, to_play=to_play, max_visits=visits,
                  komi=komi)["turns"][0]
    mis = move_infos(t, 8)
    if not mis:
        raise RuntimeError("引擎没给出任何候选——先核对棋形识别是否合法")
    lead0 = float(t.get("rootInfo", {}).get("scoreLead",
                                            mis[0].get("scoreLead", 0.0)))
    return t, mis, lead0


def _rollout(eng, size, black, white, to_play, first, depth, visits, komi,
             tough):
    """从局面推演到第 depth 手，返回 (steps, (cur_b, cur_w), next_player)。

    **每一手都独立重新搜索**，不沿上一次搜索的 PV 尾部走（实测见 _pick 注释：
    PV 第 3 手往后与实算经常不一致）。first 是已确定并验证过的第 1 手 step，
    传 None 表示从原始局面开始逐手自己选。
    """
    board = GoBoard(size[0], size[1], black, white)
    steps, pl = [], to_play
    if first is not None:
        board.play(first["color"], *first["xy"])
        steps.append(dict(first))
        pl = "W" if first["color"] == "B" else "B"
    cur_b, cur_w = board.stones()
    while len(steps) < depth:
        t = eng.query(size, cur_b, cur_w, to_play=pl, max_visits=visits,
                      komi=komi)["turns"][0]
        mis = move_infos(t, 8)
        if not mis:
            break
        mv, lead, crit, vis = _pick(mis, pl, tough)
        if mv == "pass":
            break
        x, y = from_gtp(mv)
        if not (0 <= x < size[0] and 0 <= y < size[1]):
            break
        removed = board.play(pl, x, y)
        cur_b, cur_w = board.stones()
        prev = steps[-1]["lead"] if steps else lead
        steps.append({"seq": len(steps) + 1, "color": pl, "move": mv,
                      "xy": (x, y), "black": cur_b, "white": cur_w,
                      "lead": lead, "gain": lead - prev,
                      "captured": len(removed), "criterion": crit,
                      "visits": vis})
        pl = "W" if pl == "B" else "B"
    return steps, (cur_b, cur_w), pl


def _finish(eng, size, cur_b, cur_w, pl, steps, visits, komi, pv, lead0,
            cands, kind="local", sol_index=1, n_solutions=1, alt=(),
            tough=True):
    """收尾：最终目差 + 各类提示，组装成一份 res。"""
    if steps:
        tf = eng.query(size, cur_b, cur_w, to_play=pl,
                       max_visits=max(visits // 2, 500), komi=komi)["turns"][0]
        final_lead = float(tf.get("rootInfo", {}).get("scoreLead",
                                                      steps[-1]["lead"]))
    else:
        final_lead = lead0

    alerts = []
    # 第 1 手不参与对照：多解时第 2 个解本来就不是 PV 首选，算不上"改判"
    drift = [s for s in steps
             if pv and s["seq"] > 1 and s["seq"] - 1 < len(pv)
             and pv[s["seq"] - 1] != s["move"]]
    if drift:
        # 合并成一条：根 PV 的后半段本就不可信，逐条报警会刷屏
        alerts.append(
            "第 %s 手与根节点 PV 不同（PV 尾部访问量低，逐步独立搜索后改判：%s）"
            % ("、".join(str(s["seq"]) for s in drift),
               " ".join("第%d手 %s→%s" % (s["seq"], pv[s["seq"] - 1], s["move"])
                        for s in drift)))
    if any(s["criterion"] != "引擎首选" for s in steps):
        alerts.append("变化里有手是按「目差最顽强」选的（图中标 *），"
                      "与引擎首选不同——讲题时留意这是最大抵抗而非唯一应手")
    if any(s["captured"] for s in steps):
        alerts.append("变化过程中有提子，对照棋盘时留意被提掉的棋子")
    if kind == "global":
        alerts.append("这是实战全局局面，给的是**当前最大的一手**，"
                      "不存在死活题那种唯一正解")
    return {"best": steps[0]["move"] if steps else "",
            "lead": steps[0]["lead"] if steps else lead0,
            "lead0": lead0, "final_lead": final_lead, "cands": cands,
            "steps": steps, "pv": pv, "alerts": alerts, "tough": tough,
            "kind": kind, "sol_index": sol_index,
            "n_solutions": n_solutions, "alt": list(alt)}


def solve(cols, rows, black, white, to_play="B", depth=4, visits=3000,
          komi=None, tough=True, kind="local"):
    """单解版：求正解与变化序列。

    kind: "local"（死活/手筋题，出完整变化）| "global"（实战全局题，只出第 1 手）
    """
    komi = _default_komi(cols, rows) if komi is None else komi
    eng = get_engine()
    size = (cols, rows)
    depth = 1 if kind == "global" else depth

    _, mis0, lead0 = _root(eng, size, black, white, to_play, visits, komi)
    cands = [(mi["move"], round(float(mi.get("scoreLead", 0)), 1))
             for mi in mis0[:6]]
    pv = list(mis0[0].get("pv", []))
    steps, (cb, cw), pl = _rollout(eng, size, black, white, to_play, None,
                                   depth, visits, komi, tough)
    return _finish(eng, size, cb, cw, pl, steps, visits, komi, pv, lead0,
                   cands, kind=kind, tough=tough)


def solve_multi(cols, rows, black, white, to_play="B", depth=5, visits=3000,
                komi=None, tough=True, tol=1.0, topk=4):
    """**多解版**：把引擎眼里等价的正解都找出来，每个正解各出一套变化。

    为什么要单独一套（2026-09-11 用户要求）：死活/手筋题存在双解，只给一个
    "正解"会让孩子以为另外那手是错的。

    判定办法：对根搜索的前 topk 个候选，**逐个落子后独立做一次同深度的搜索**，
    用走完该手后的 scoreLead（黑方视角）横向比较，与最优手相差 <= tol 目的
    都算正解。逐个独立验证而不是直接比根节点 moveInfos 的 scoreLead，是因为
    后者访问量悬殊（实测 vis=1 的候选能报出离谱目差）。

    返回 {"solutions": [res, ...], "cands", "lead0", "n_solutions", ...}
    """
    komi = _default_komi(cols, rows) if komi is None else komi
    eng = get_engine()
    size = (cols, rows)

    _, mis0, lead0 = _root(eng, size, black, white, to_play, visits, komi)
    pv = list(mis0[0].get("pv", []))

    # 1) 逐个候选独立验证：落子后重新搜索，拿该局面的目差
    tried = []
    for mi in mis0[:topk]:
        mv = mi["move"]
        if mv == "pass":
            continue
        x, y = from_gtp(mv)
        if not (0 <= x < cols and 0 <= y < rows):
            continue
        board = GoBoard(cols, rows, black, white)
        removed = board.play(to_play, x, y)
        nb, nw = board.stones()
        other = "W" if to_play == "B" else "B"
        t = eng.query(size, nb, nw, to_play=other, max_visits=visits,
                      komi=komi)["turns"][0]
        lead = float(t.get("rootInfo", {}).get("scoreLead",
                                               mi.get("scoreLead", 0)))
        tried.append({"move": mv, "lead": lead, "xy": (x, y),
                      "captured": len(removed), "visits": int(mi.get("visits", 0)),
                      "criterion": "引擎首选", "color": to_play,
                      "black": nb, "white": nw, "seq": 1,
                      "gain": lead - lead0})
    if not tried:
        raise RuntimeError("引擎没给出任何可落子的候选")
    cands = [(d["move"], round(d["lead"], 1)) for d in tried]

    # 2) 与最优手相差 tol 目以内的都算正解
    best_lead = max(d["lead"] for d in tried) if to_play == "B" \
        else min(d["lead"] for d in tried)
    sols = sorted([d for d in tried if abs(d["lead"] - best_lead) <= tol],
                  key=lambda d: -d["lead"] if to_play == "B" else d["lead"])

    # 3) 每个正解各推演一套变化
    out = []
    for i, s in enumerate(sols):
        alt = [(d["move"], round(d["lead"], 1)) for j, d in enumerate(sols)
               if j != i]
        if depth <= 1:
            steps, (cb, cw), pl = [s], (s["black"], s["white"]), \
                ("W" if to_play == "B" else "B")
        else:
            steps, (cb, cw), pl = _rollout(eng, size, black, white, to_play,
                                           s, depth, visits, komi, tough)
        res = _finish(eng, size, cb, cw, pl, steps, visits, komi, pv, lead0,
                      cands, kind="local", sol_index=i + 1,
                      n_solutions=len(sols), alt=alt, tough=tough)
        out.append(res)
    return {"solutions": out, "cands": cands, "lead0": lead0,
            "n_solutions": len(sols), "best": sols[0]["move"],
            "lead": sols[0]["lead"], "kind": "local",
            "alerts": ([]
                       if len(sols) == 1 else
                       ["本题有 %d 个正解：%s，彼此相差 %.1f 目以内 "
                        "—— 都是对的，别只认一个"
                        % (len(sols), " / ".join(d["move"] for d in sols),
                           abs(sols[0]["lead"] - sols[-1]["lead"]))])}


def explain(res, to_play="B"):
    """把求解结果讲成一句人话（讲题卡底部用）。"""
    if not res["steps"]:
        return ["引擎未给出变化，无法讲解"]
    s1 = res["steps"][0]
    cn = "黑" if s1["color"] == "B" else "白"
    n_sol = res.get("n_solutions", 1)
    out = []
    if res.get("kind") == "global":
        out.append("当前最大一手：第1手 %s %s，落子后黑方领先 %.1f 目。"
                   % (cn, s1["move"], s1["lead"]))
        if len(res["cands"]) >= 2:
            out.append("次选 %s（%.1f 目），与这手差 %.1f 目。"
                       % (res["cands"][1][0], res["cands"][1][1],
                          abs(res["cands"][0][1] - res["cands"][1][1])))
        out.append("全局题是实战局面，这一手是当前最大的一手，不是唯一答案。")
        return out
    if n_sol > 1:
        out.append("正解 %d/%d：第1手 %s %s，落子后黑方领先 %.1f 目。"
                   % (res.get("sol_index", 1), n_sol, cn, s1["move"],
                      s1["lead"]))
        if res.get("alt"):
            out.append("另一正解 %s（%.1f 目），与本手只差 %.1f 目 —— "
                       "两解都对，都要会。"
                       % ("、".join("%s" % m for m, _ in res["alt"]),
                          res["alt"][0][1],
                          abs(res["alt"][0][1] - s1["lead"])))
    else:
        out.append("正解：第1手 %s %s，落子后黑方领先 %.1f 目。"
                   % (cn, s1["move"], s1["lead"]))
        if len(res["cands"]) >= 2:
            out.append("次选 %s（%.1f 目），与正解差 %.1f 目。"
                       % (res["cands"][1][0], res["cands"][1][1],
                          abs(res["cands"][0][1] - res["cands"][1][1])))
    if len(res["steps"]) >= 2:
        s2 = res["steps"][1]
        tag = "" if s2.get("criterion", "引擎首选") == "引擎首选" \
            else "（按「目差最顽强」选出，非引擎第一选择）"
        out.append("对手最强抵抗是第2手 %s %s，之后黑方领先 %.1f 目%s。"
                   % ("黑" if s2["color"] == "B" else "白", s2["move"],
                      s2["lead"], tag))
    out.append("想一想：第1手为什么走这里？如果换个位置，对手会怎么反击？")
    return out


def variation_single(title, cols, rows, corner, black0, white0, res,
                     to_play="B"):
    """**一张**棋盘画完整条变化——传统棋书「参考图」的形式。

    盘面取**走完最后一手**的棋形，每手棋子的正中央写序号 1..N（2026-09-11
    用户定稿：不要拆成 N 张小图）。右侧图例列每手的落子方、坐标与该手后的
    目差；按「目差最顽强」选出的手标 *，提示它并非引擎第一选择。
    被提掉的子不再画，序号落在空交叉点上时用虚线环标出该手曾落在此处。
    """
    steps = res["steps"]
    if steps:
        fb, fw, last_xy = steps[-1]["black"], steps[-1]["white"], steps[-1]["xy"]
    else:
        fb, fw, last_xy = black0, white0, None
    nums = [(s["seq"], s["xy"][0], s["xy"][1]) for s in steps]

    big = cols >= 15
    cell = 30 if big else 38
    margin = 34 if big else 42
    sub = ("原题 黑 %d · 白 %d ｜ 共 %d 手 ｜ 走完后黑方领先 %.1f 目"
           % (len(black0), len(white0), len(steps),
              res.get("final_lead", res["lead"])))
    bw, bh, inner = board_parts(title, fb, fw, corner, cols, rows,
                                cell, margin, nums=nums, last=last_xy, sub=sub)

    head_h = 92
    pad = 40
    legend_w = 430
    row_h = 38
    notes = explain(res, to_play)
    note_h = 26 * (len(notes) + len(res.get("alerts", []))) + 30
    board_top = head_h + 14
    legend_h = 56 + row_h * (len(steps) + 1)
    W = pad + bw + 34 + legend_w + pad
    H = board_top + max(bh, legend_h) + note_h + 26

    def txt(x, y, s, size=15, color=C_TEXT, bold=False):
        w = ' font-weight="600"' if bold else ""
        return (f'<text x="{x}" y="{y}" font-size="{size}"{w} fill="{color}" '
                f'font-family="PingFang SC, sans-serif">{html.escape(s)}</text>')

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:100%;background:{C_BG}">']
    p.append(txt(pad, 46, title, 22, C_TEXT, True))
    if res.get("kind") == "global":
        sub2 = ("最大一手 %s ｜ 目差均为黑方视角，起手领先 %.1f 目"
                % (res["best"], res["lead"]))
    elif res.get("n_solutions", 1) > 1:
        sub2 = ("正解 %d/%d：%s ｜ %d 步变化 ｜ 起手领先 %.1f 目"
                % (res.get("sol_index", 1), res["n_solutions"], res["best"],
                   len(steps), res["lead"]))
    else:
        sub2 = ("正解 第1手 %s ｜ %d 步变化 ｜ 目差均为黑方视角，起手领先 %.1f 目"
                % (res["best"], len(steps), res["lead"]))
    p.append(txt(pad, 74, sub2, 15, C_SUB))

    p.append(f'<g transform="translate({pad},{board_top})">{inner}</g>')

    # 右侧图例
    lx = pad + bw + 34
    ly = board_top + 10
    p.append(txt(lx, ly, "变化顺序", 16, C_TEXT, True))
    ly += 34
    for s in steps:
        tail = ""
        if s["captured"]:
            tail += " ｜ 提 %d 子" % s["captured"]
        if s.get("criterion", "引擎首选") != "引擎首选":
            tail += " *"
        if s.get("visits", 0) < 300:      # 搜索不够深，目差估值不稳，标出来
            tail += "（仅%d次访问）" % s["visits"]
        p.append(f'<circle cx="{lx + 13}" cy="{ly - 5}" r="12" '
                 f'fill="{C_MARK}"/>')
        p.append(f'<text x="{lx + 13}" y="{ly - 5}" font-size="14" '
                 f'font-weight="700" fill="#FFFFFF" text-anchor="middle" '
                 f'dominant-baseline="central" '
                 f'font-family="PingFang SC, sans-serif">{s["seq"]}</text>')
        # 目差一律黑方视角（KataGo scoreLead 就是这样，见文件头）
        p.append(txt(lx + 34, ly,
                     "%s %s ｜ 领先 %+.1f 目%s"
                     % ("黑" if s["color"] == "B" else "白", s["move"],
                        s["lead"], tail)))
        ly += row_h
    if steps:
        p.append(txt(lx, ly + 4,
                     "* = 按「目差最顽强」选出（非引擎第一选择）"
                     if any(s.get("criterion", "引擎首选") != "引擎首选"
                            for s in steps)
                     else "每手均取引擎最优（白方＝最顽强抵抗）",
                     13, C_SUB))

    # 底部讲解
    ny = board_top + max(bh, legend_h) + 34
    for line in notes:
        p.append(txt(pad, ny, line, 15, C_TEXT))
        ny += 26
    for a in res.get("alerts", []):
        p.append(txt(pad, ny, "注意：" + a, 14, C_MARK))
        ny += 26
    p.append("</svg>")
    return "".join(p)
