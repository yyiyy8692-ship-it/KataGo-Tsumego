"""批改与讲解：正解验证、孩子答案三层比对、一致性警报、启发阶梯文案。

教训固化（勿回退）：
- 先后手以用户确认的题型为准（默认黑先），系统不自己假设
- forced moves 一律用 gtp() 生成（引擎层已强制）
- 引擎 vs 孩子/题面矛盾 → 标"存疑"警报，第一怀疑对象是识别，不是孩子
- 比对三层分开：① 首着要点 ② 结论（杀/活/劫）③ 后续次序
"""
from engine.katago import get_engine, best_moves, gtp, from_gtp

# 题型 -> (谁下, 目标方, 目标描述)
PROBLEM_TYPES = {
    "B_kill": ("B", "W", "黑先杀白"),
    "B_live": ("B", "B", "黑先做活"),
    "W_kill": ("W", "B", "白先杀黑"),
    "W_live": ("W", "W", "白先做活"),
}


def _verdict_from_ownership(ownership, cols, rows, target_stones, to_play_sign):
    """ownership 正值=黑。目标棋群平均归属 >0.6 判活（归自己），<-0.6 判死。"""
    if not ownership or not target_stones:
        return "未知", 0.0
    vals = []
    for x, y in target_stones:
        if 0 <= x < cols and 0 <= y < rows:
            vals.append(ownership[y * cols + x])
    if not vals:
        return "未知", 0.0
    avg = sum(vals) / len(vals)
    # 统一成"目标方视角"：目标方是白则取反
    v = avg if to_play_sign == "B" else -avg
    if v > 0.6:
        return "活", v
    if v < -0.6:
        return "死", v
    return "不明", v


def grade(cols, rows, black, white, ptype, kid_moves, note=""):
    """kid_moves: [(seq,"B"/"W",x,y)] 家长确认后的孩子变化线。
    返回结构化批改结果。"""
    to_play, target_color, type_desc = PROBLEM_TYPES[ptype]
    target_stones = white if target_color == "W" else black
    eng = get_engine()

    # 1) 正解：初始局面引擎分析
    root = eng.query((cols, rows), black, white, to_play=to_play, max_visits=6000)
    t0 = root["turns"][0]
    cands = best_moves(t0, 6)
    sol_move, sol_lead, sol_pv = cands[0]

    # 2) 沿引擎 PV 走到定型，判目标棋群死活
    pv_moves = []
    pl = to_play
    for m in sol_pv:
        pv_moves.append((pl, m))
        pl = "W" if pl == "B" else "B"
        if len(pv_moves) >= 12 or (len(pv_moves) >= 2 and pv_moves[-1][1] == "pass" and pv_moves[-2][1] == "pass"):
            break
    pv_res = eng.query((cols, rows), black, white, moves=pv_moves,
                       to_play=to_play, max_visits=1000)
    final_t = pv_res["turns"][len(pv_moves)]
    verdict, vscore = _verdict_from_ownership(
        final_t.get("ownership"), cols, rows, target_stones, target_color)

    # 3) 让先测试（无条件死活裁决）：目标方对手连走两手也杀不掉 = 无条件活
    pass_first = eng.query((cols, rows), black, white,
                           moves=[(to_play, "pass")], to_play=to_play, max_visits=2000)
    # 4) 孩子变化线逐手验证
    kid_report = []
    if kid_moves:
        moves_so_far = []
        pl = to_play
        for seq, color, x, y in sorted(kid_moves):
            # 当前局面引擎首选
            cur = eng.query((cols, rows), black, white, moves=moves_so_far,
                            to_play=to_play, max_visits=3000)
            ct = cur["turns"][len(moves_so_far)]
            top = best_moves(ct, 5)
            mv = gtp(x, y)
            rank = next((i + 1 for i, (m, _, _) in enumerate(top) if m == mv), None)
            best_mv, best_lead, _ = top[0]
            kid_lead = next((l for m, l, _ in top if m == mv), None)
            kid_report.append({
                "seq": seq, "move": mv, "engine_best": best_mv,
                "rank": rank, "is_best": mv == best_mv,
                "lead_gap": round(abs((kid_lead or best_lead) - best_lead), 1) if rank else None,
            })
            moves_so_far.append((color, mv))
        # 变化线终点死活
        end = eng.query((cols, rows), black, white, moves=moves_so_far,
                        to_play=to_play, max_visits=2000)
        end_t = end["turns"][len(moves_so_far)]
        kid_verdict, _ = _verdict_from_ownership(
            end_t.get("ownership"), cols, rows, target_stones, target_color)
    else:
        kid_verdict = None

    # 5) 一致性警报（触发 → 存疑，建议复查棋形）
    alerts = []
    if verdict == "不明":
        alerts.append("引擎无法判定净死/净活（可能涉劫或识别误差），建议核对棋形")
    if kid_verdict and verdict != "不明" and kid_verdict != verdict:
        alerts.append(f"孩子变化线结局（{kid_verdict}）与引擎正解结局（{verdict}）不一致——"
                      "先核对棋形识别与手顺录入，再判对错")
    if note and ("劫" in note) and verdict in ("活", "死"):
        alerts.append(f"题面/孩子提到「劫」但引擎判净{verdict}——识别差一子即可翻转劫↔净，"
                      "请优先复查棋形，不要直接判错")

    return {
        "type_desc": type_desc,
        "solution": {"move": sol_move, "pv": sol_pv[:8], "verdict": verdict},
        "candidates": [(m, l) for m, l, _ in cands[:4]],
        "kid_report": kid_report,
        "kid_verdict": kid_verdict,
        "alerts": alerts,
    }


HINT_LADDER = {
    "kill": ["先看看：这块棋要做活，需要几只眼？现在有几个眼位？",
             "数一数它内部有几个空点，是什么形状？（直三、曲四、板六…）",
             "这种形状，要点在哪里？腰线/中心通常是要害。",
             "下在要点后，想想对方最强的抵抗是什么，你还能杀吗？"],
    "live": ["先看看：这块棋现在有几个确定的眼？还差几只？",
             "数一数内部空点的形状，哪里是能长出眼来的关键交叉点？",
             "要点往往同时也是对方杀棋的要点——谁先抢到谁赢。",
             "占到要点后，对方最狠的破眼手段是什么，你如何应对？"],
}


def hints_for(ptype):
    kind = "kill" if "kill" in ptype else "live"
    return HINT_LADDER[kind]
