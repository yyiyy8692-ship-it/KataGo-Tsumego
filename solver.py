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


def solve(cols, rows, black, white, to_play="B", depth=4, visits=3000,
          komi=None, tough=True):
    """求正解与变化序列。

    **每一手都独立重新搜索**（depth 次查询），不沿上一次搜索的 PV 尾部走。
    原因（实测 diag_bestmove.py，2026-09-11）：手筋题3 的第 4 手，根节点 PV
    给 F6，而落子后重新搜索的最优选是 C3 —— PV 越往尾部访问量越低、越不可信。
    KaTrain / Lizzie 的教学模式同样是逐步重算，不是复用 PV。

    tough: 是否启用「最顽强抵抗」目差校验（详见 _pick）。
    depth: 变化图要几手（3~5）。返回 dict：
        best/lead/cands: 初始局面引擎首选、黑方领先目差、前几个候选
        steps: [{"seq","color","move","xy","black","white","lead","gain",
                 "captured","criterion","visits"}...]
        lead0/final_lead: 初始局面 / 走完 depth 手后黑方领先
        pv: 根节点 PV（用于与逐步搜索结果对照，drift 会进 alerts）
        alerts: 需要人工留意的提示
    """
    komi = _default_komi(cols, rows) if komi is None else komi
    eng = get_engine()
    size = (cols, rows)

    board = GoBoard(cols, rows, black, white)
    cur_b, cur_w = board.stones()
    pl = to_play
    steps = []

    # 1) 根查询：候选、PV、初始目差
    t = eng.query(size, cur_b, cur_w, to_play=pl, max_visits=visits,
                  komi=komi)["turns"][0]
    mis0 = move_infos(t, 8)
    if not mis0:
        raise RuntimeError("引擎没给出任何候选——先核对棋形识别是否合法")
    cands = [(mi["move"], round(float(mi.get("scoreLead", 0)), 1))
             for mi in mis0[:6]]
    best = mis0[0]["move"]
    best_lead = float(mis0[0].get("scoreLead", 0.0))
    lead0 = float(t.get("rootInfo", {}).get("scoreLead", best_lead))
    pv = list(mis0[0].get("pv", []))

    # 2) 逐手独立搜索
    for i in range(depth):
        if i:
            t = eng.query(size, cur_b, cur_w, to_play=pl, max_visits=visits,
                          komi=komi)["turns"][0]
            mis = move_infos(t, 8)
            if not mis:
                break
        else:
            mis = mis0
        mv, lead, crit, vis = _pick(mis, pl, tough)
        if mv == "pass":
            break
        x, y = from_gtp(mv)
        if not (0 <= x < cols and 0 <= y < rows):
            break
        removed = board.play(pl, x, y)
        prev_lead = steps[-1]["lead"] if steps else lead0
        cur_b, cur_w = board.stones()
        steps.append({"seq": i + 1, "color": pl, "move": mv, "xy": (x, y),
                      "black": cur_b, "white": cur_w, "lead": lead,
                      "gain": lead - prev_lead, "captured": len(removed),
                      "criterion": crit, "visits": vis})
        pl = "W" if pl == "B" else "B"

    # 3) 收尾：走完后局面的目差（独立评估一次，比沿用最后一手的估值稳）
    if steps:
        tf = eng.query(size, cur_b, cur_w, to_play=pl,
                       max_visits=max(visits // 2, 500), komi=komi)["turns"][0]
        final_lead = float(tf.get("rootInfo", {}).get("scoreLead",
                                                      steps[-1]["lead"]))
    else:
        final_lead = lead0

    # 4) 提示：双解、提子、PV 与逐步搜索不一致
    alerts = []
    if len(cands) >= 2 and abs(cands[0][1] - cands[1][1]) < 1.0:
        alerts.append(
            "首选 %s 与次选 %s 只差 %.1f 目 —— 引擎眼里接近双解，"
            "别把它当唯一正解" % (cands[0][0], cands[1][0],
                                abs(cands[0][1] - cands[1][1])))
    drift = [s for s in steps
             if pv and s["seq"] - 1 < len(pv) and pv[s["seq"] - 1] != s["move"]]
    if drift:
        # 合并成一条：根目录 PV 的后半段本就不可信，逐条报警会刷屏
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
    return {"best": best, "lead": best_lead, "lead0": lead0,
            "final_lead": final_lead, "cands": cands, "steps": steps,
            "pv": pv, "alerts": alerts, "tough": tough}


def explain(res, to_play="B"):
    """把求解结果讲成一句人话（讲题卡底部用）。"""
    if not res["steps"]:
        return ["引擎未给出变化，无法讲解"]
    s1 = res["steps"][0]
    out = ["正解：第1手 %s %s，落子后黑方领先 %.1f 目。"
           % ("黑" if s1["color"] == "B" else "白", s1["move"], s1["lead"])]
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
    p.append(txt(pad, 74,
                 "正解 第1手 %s ｜ %d 步变化 ｜ 目差均为黑方视角，起手领先 %.1f 目"
                 % (res["best"], len(steps), res["lead"]), 15, C_SUB))

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
