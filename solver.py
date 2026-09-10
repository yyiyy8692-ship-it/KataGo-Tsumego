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
from engine.katago import get_engine, best_moves, gtp, from_gtp


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


def solve(cols, rows, black, white, to_play="B", depth=4, visits=3000,
          komi=None):
    """求正解与变化序列。

    depth: 变化图要几手（3~5）。返回 dict：
        best/lead/cands: 初始局面引擎首选、黑方领先目差、前几个候选
        steps: [{"seq","color","move","xy","black","white","lead","gain",
                 "captured"}, ...]，black/white 是该手**落下后**的局面
        lead0: 初始局面黑方领先
        alerts: 需要人工留意的提示
    """
    komi = _default_komi(cols, rows) if komi is None else komi
    eng = get_engine()

    # 1) 初始局面：拿首选与 PV
    root = eng.query((cols, rows), black, white, to_play=to_play,
                     max_visits=visits, komi=komi)
    t0 = root["turns"][0]
    cands = best_moves(t0, 6)
    if not cands:
        raise RuntimeError("引擎没给出任何候选——先核对棋形识别是否合法")
    best, best_lead, pv = cands[0]
    lead0 = float(t0.get("rootInfo", {}).get("scoreLead", best_lead))

    # 2) 取前 depth 手，推演每手后的局面
    seq_moves, plan = [], []
    board = GoBoard(cols, rows, black, white)
    pl = to_play
    for mv in pv[:depth]:
        if mv == "pass":
            break
        x, y = from_gtp(mv)
        if not (0 <= x < cols and 0 <= y < rows):
            break
        seq_moves.append((pl, mv))
        plan.append((pl, mv, x, y))
        pl = "W" if pl == "B" else "B"
    if not seq_moves:
        return {"best": best, "lead": best_lead, "lead0": lead0,
                "cands": [(m, l) for m, l, _ in cands], "steps": [],
                "alerts": ["引擎 PV 为空，无法生成变化图"]}

    # 3) 一次查询拿各手的评估：turns[k] = 前 k 手走完后的局面
    res = eng.query((cols, rows), black, white, moves=seq_moves,
                    to_play=to_play, max_visits=max(visits // 2, 500),
                    komi=komi)
    leads = []
    for k in range(len(seq_moves) + 1):
        t = res["turns"].get(k)          # 缺 turn 时不崩，退化成 0
        leads.append(float(t.get("rootInfo", {}).get("scoreLead", 0.0))
                     if t else 0.0)

    steps = []
    board2 = GoBoard(cols, rows, black, white)
    for i, (pl_i, mv, x, y) in enumerate(plan):
        removed = board2.play(pl_i, x, y)
        bl, wh = board2.stones()
        steps.append({
            "seq": i + 1, "color": pl_i, "move": mv, "xy": (x, y),
            "black": bl, "white": wh,
            "lead": leads[i + 1], "gain": leads[i + 1] - leads[i],
            "captured": len(removed),
        })

    # 4) 双解提示：引擎自己都分不清时不该宣称"唯一正解"
    alerts = []
    if len(cands) >= 2 and abs(cands[0][1] - cands[1][1]) < 1.0:
        alerts.append(
            "首选 %s 与次选 %s 只差 %.1f 目 —— 引擎眼里接近双解，"
            "别把它当唯一正解" % (cands[0][0], cands[1][0],
                                abs(cands[0][1] - cands[1][1])))
    if any(s["captured"] for s in steps):
        alerts.append("变化过程中有提子，对照棋盘时留意被提掉的棋子")
    return {"best": best, "lead": best_lead, "lead0": lead0,
            "cands": [(m, l) for m, l, _ in cands], "steps": steps,
            "alerts": alerts}


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
        out.append("对手最强抵抗是第2手 %s %s，之后黑方领先 %.1f 目。"
                   % ("黑" if s2["color"] == "B" else "白", s2["move"],
                      s2["lead"]))
    out.append("想一想：第1手为什么走这里？如果换个位置，对手会怎么反击？")
    return out


def variation_sheet(title, cols, rows, corner, black0, white0, res,
                    to_play="B", with_original=True):
    """把「原题 + N 手变化」拼成一张 SVG。

    整盘（>=15 路）每排放 2 张，手筋题一排 5 张以内；每张棋盘高亮最新一手
    并标序号，副行写该手的目差与收益。
    """
    steps = res["steps"]
    boards = []                                   # [(name, black, white, mark, seq, sub)]
    if with_original:
        boards.append(("原题", black0, white0, None, None,
                       "黑 %d · 白 %d ｜ 黑方领先 %.1f 目"
                       % (len(black0), len(white0), res["lead0"])))
    for s in steps:
        sub = "%s %s ｜ 领先 %.1f 目" % ("黑" if s["color"] == "B" else "白",
                                        s["move"], s["lead"])
        if s["captured"]:
            sub += "（提 %d 子）" % s["captured"]
        boards.append(("第 %d 手" % s["seq"], s["black"], s["white"],
                       s["xy"], s["seq"], sub))

    big = cols >= 15
    cell = 26 if big else 34
    margin = 30 if big else 36
    bw, bh, _ = board_parts("x", black0, white0, corner, cols, rows,
                            cell, margin)
    per_row = 2 if big else min(len(boards), 5)
    gap = 28
    n_row = (len(boards) + per_row - 1) // per_row

    head_h = 96
    note_h = 40 + 26 * len(res.get("alerts", [])) + 26 * 4
    W = 40 * 2 + per_row * bw + (per_row - 1) * gap
    H = head_h + 20 + n_row * bh + (n_row - 1) * gap + note_h + 30

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:100%;background:{C_BG}">']
    p.append(f'<text x="40" y="46" font-size="22" font-weight="600" '
             f'fill="{C_TEXT}" font-family="PingFang SC, sans-serif">'
             f'{html.escape(title)}</text>')
    p.append(f'<text x="40" y="74" font-size="15" fill="{C_SUB}" '
             f'font-family="PingFang SC, sans-serif">'
             f'正解 第1手 {html.escape(res["best"])} ｜ '
             f'{len(steps)} 步变化 ｜ 黑方领先 {res["lead"]:.1f} 目</text>')

    y0 = head_h + 20
    for i, (name, bl, wh, mark, seq, sub) in enumerate(boards):
        r, c = divmod(i, per_row)
        x = 40 + c * (bw + gap)
        y = y0 + r * (bh + gap)
        _, _, inner = board_parts(name, bl, wh, corner, cols, rows,
                                  cell, margin, mark, seq, sub)
        p.append(f'<g transform="translate({x},{y})">{inner}</g>')

    ny = y0 + n_row * bh + (n_row - 1) * gap + 34
    for line in explain(res, to_play):
        p.append(f'<text x="40" y="{ny}" font-size="15" fill="{C_TEXT}" '
                 f'font-family="PingFang SC, sans-serif">'
                 f'{html.escape(line)}</text>')
        ny += 26
    for a in res.get("alerts", []):
        p.append(f'<text x="40" y="{ny}" font-size="14" fill="{C_MARK}" '
                 f'font-family="PingFang SC, sans-serif">'
                 f'注意：{html.escape(a)}</text>')
        ny += 26
    p.append("</svg>")
    return "".join(p)
