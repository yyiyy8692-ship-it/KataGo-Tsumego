"""诊断：引擎「最强应手」到底按什么排序选出来的？

核心问题（2026-09-11）：
  变化图里白棋的应手，现在是直接沿用根节点搜索 PV 的第 2、4 手。PV 尾部访问
  量远低于根节点，且 KataGo 的 moveInfos 顺序是按 utility（胜率+目差混合），
  不是按 scoreLead。这两点都会让"最强应手"讲错。

本脚本做三组对照：
  A. 根节点：引擎首选 vs 按 scoreLead 排序的榜首，是否同一手
  B. 走完第1手后白方：引擎首选 vs scoreLead 最小（= 对黑最不利、最顽强），是否同一手
  C. 根节点 PV 的第 2/3 手 vs 每手独立重新搜索的结果，是否一致

用法： python diag_bestmove.py 3    # 第4题（手筋题1）
"""
import sys

sys.path.insert(0, ".")

import solve_page
import solver
from engine.katago import get_engine, gtp, from_gtp


def show(tag, turn, topn=6):
    mis = turn.get("moveInfos", [])[:topn]
    print("  [%s] rootLead=%s" % (tag, turn.get("rootInfo", {}).get("scoreLead")))
    for i, mi in enumerate(mis):
        print("    %d. %-4s lead=%+7.2f win=%.3f util=%+.4f vis=%-6d pv=%s"
              % (i + 1, mi["move"], mi.get("scoreLead", 0), mi.get("winrate", 0),
                 mi.get("utility", 0), mi.get("visits", 0),
                 " ".join(mi.get("pv", [])[:4])))
    return mis


def main(idx=3, visits=3000):
    names = solve_page.DEFAULT_NAMES
    parts = ["photos/paper_p1_300_b%d.png" % i for i in range(len(names))]
    parts = [p for p in parts if __import__("os").path.exists(p)]
    p = parts[idx]
    cols, rows, black, white, c = solve_page.recognize(p)
    print("=== 第%d题 %s  %dx%d 黑%d 白%d corner=%s"
          % (idx + 1, names[idx], cols, rows, len(black), len(white), c))
    komi = solver._default_komi(cols, rows)
    eng = get_engine()

    # A) 根节点
    root = eng.query((cols, rows), black, white, to_play="B",
                     max_visits=visits, komi=komi)
    t0 = root["turns"][0]
    mis0 = show("根·黑先", t0)
    by_lead_b = min(mis0, key=lambda m: m["scoreLead"])   # 黑要最大 -> 白视角最小
    best_b = max(mis0, key=lambda m: m["scoreLead"])
    print("    -> 引擎首选 %s ；按 scoreLead 最大 %s ；两者%s"
          % (mis0[0]["move"], best_b["move"],
             "一致" if mis0[0]["move"] == best_b["move"] else "**不一致**"))
    b1 = mis0[0]["move"]
    pv = mis0[0].get("pv", [])
    print("    -> 根节点 PV: %s" % " ".join(pv[:6]))

    # B) 走完第1手，白方独立搜索
    x, y = from_gtp(b1)
    board = solver.GoBoard(cols, rows, black, white)
    board.play("B", x, y)
    bl, wh = board.stones()
    r2 = eng.query((cols, rows), bl, wh, to_play="W", max_visits=visits,
                   komi=komi)
    t1 = r2["turns"][0]
    mis1 = show("第1手后·白先", t1)
    tough_w = min(mis1, key=lambda m: m["scoreLead"])     # 让黑领先最少 = 最顽强
    print("    -> 引擎首选 %s ；按 scoreLead 最小(最顽强) %s ；两者%s"
          % (mis1[0]["move"], tough_w["move"],
             "一致" if mis1[0]["move"] == tough_w["move"] else "**不一致**"))
    print("    -> 根节点 PV 第2手 = %s" % (pv[1] if len(pv) > 1 else "(无)"))

    # C) 每手独立搜索 vs PV 尾部
    print("  [逐步独立搜索]")
    bb = solver.GoBoard(cols, rows, black, white)
    pl = "B"
    seq = [(pl, b1)]
    for k in range(4):
        if k == 0:
            continue
        pass
    # 逐步：先落 b1，再连查 4 手
    bcur = solver.GoBoard(cols, rows, black, white)
    bcur.play("B", *from_gtp(b1))
    cbl, cwh = bcur.stones()
    pl = "W"
    for k in range(4):
        r = eng.query((cols, rows), cbl, cwh, to_play=pl, max_visits=visits,
                      komi=komi)
        mk = r["turns"][0]["moveInfos"]
        if not mk:
            break
        pick = mk[0]["move"]
        pick_lead = min(mk, key=lambda m: m["scoreLead"])["move"] if pl == "W" \
            else max(mk, key=lambda m: m["scoreLead"])["move"]
        print("    第%d手(%s): 引擎首选 %-4s | 目差准则 %-4s | PV尾部 %-4s %s"
              % (k + 2, pl, pick, pick_lead,
                 pv[k + 1] if len(pv) > k + 1 else "-",
                 "" if pick == (pv[k + 1] if len(pv) > k + 1 else pick)
                 else "  <-- 与PV不同"))
        if pick == "pass":
            break
        x, y = from_gtp(pick)
        bcur.play(pl, x, y)
        cbl, cwh = bcur.stones()
        pl = "W" if pl == "B" else "B"
    return True


if __name__ == "__main__":
    i = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    v = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
    main(i, v)
