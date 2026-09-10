"""端到端冒烟：识别结果 → KataGo，用引擎输出反查棋形是否可信。

为什么要这一步：识别没有真值标注，只能靠自洽检查（黑+白+空=格数）。但自洽
不代表对——整盘错一格、黑白整体互换都能"自洽"。KataGo 是独立的第二意见：
局面合法且合理时，它给出的胜率/最佳手应当是有意义的；棋形被识别乱时，
引擎会给出荒谬输出（例如一方胜率 99%、最佳手落在明显无意义的点）。

用法：
    python e2e_katago.py            # 跑全部（打印稿 6 题 + 拍照稿 3 道整盘）

2026-09-10 首次实测结论（杨震杰老师习题 PDF，题面写明「全部黑先」）：
  9 个局面全部被引擎接受，目差 -9.1 ~ 28.8，无非法局面、无荒谬值 → 识别可信。
  手筋题需提高 visits 才能收敛：800 时手筋3 的 G6/F3 并列（11.0/10.6），
  8000 时拉开但仍只差 0.4 目 —— 批改判「唯一正解」前必须先测收敛性，
  见下方 warn 里的 gap 判读。批改用的 visits 建议 >=3000。
"""
import sys
import time

sys.path.insert(0, ".")

from recognition import detect as D
import fullboard as F
from engine.katago import get_engine, best_moves

PHOTOS = "photos/%s.png"


def recognize_block(path):
    """返回 (cols, rows, black, white, kind)。整盘优先，失败回退局部。"""
    try:
        r = F.recognize_full_board(path)
        if r and r.get("cols") and r["cols"] >= 15:
            return (r["cols"], r["rows"], list(r["black"]), list(r["white"]),
                    "整盘")
    except Exception:
        pass                      # 局部题锁不出 19x19 属正常，静默回退
    r = D.recognize(path)
    return r["cols"], r["rows"], list(r["black"]), list(r["white"]), "局部"


def ask_katago(eng, cols, rows, black, white, to_play="B", visits=800):
    """问引擎：初始胜率 + 最佳手 + 前 3 候选。komi 按棋盘大小定。"""
    komi = 7.5 if min(cols, rows) >= 15 else 0.0
    t = time.time()
    res = eng.query((cols, rows), black, white, to_play=to_play,
                    max_visits=visits, komi=komi)
    dt = time.time() - t
    t0 = res["turns"][0]
    cands = best_moves(t0, 3)
    # winrate：黑方视角，scoreLead 是目差
    lead = cands[0][1] if cands else 0.0
    return {
        "dt": dt,
        "best": cands[0][0] if cands else None,
        "lead": lead,
        "cands": [(m, l) for m, l, _ in cands],
        "pv": cands[0][2][:6] if cands else [],
    }


def main():
    eng = get_engine()
    jobs = [
        ("打印稿 第1题(全局1)", "paper_p1_300_b0"),
        ("打印稿 第2题(全局2)", "paper_p1_300_b1"),
        ("打印稿 第3题(全局3)", "paper_p1_300_b2"),
        ("打印稿 第4题(手筋1)", "paper_p1_300_b3"),
        ("打印稿 第5题(手筋2)", "paper_p1_300_b4"),
        ("打印稿 第6题(手筋3)", "paper_p1_300_b5"),
        ("拍照稿 第1题", "page1_b0"),
        ("拍照稿 第2题", "page1_b1"),
        ("拍照稿 第3题", "page1_b2"),
    ]
    for label, name in jobs:
        path = PHOTOS % name
        print("=" * 62)
        print("%s   (%s)" % (label, path))
        try:
            cols, rows, black, white, kind = recognize_block(path)
        except Exception as e:
            print("   识别失败：%s" % e)
            continue
        print("   %s %dx%d  黑%d 白%d 空%d"
              % (kind, cols, rows, len(black), len(white),
                 cols * rows - len(black) - len(white)))
        try:
            k = ask_katago(eng, cols, rows, black, white)
        except Exception as e:
            print("   引擎失败：%s" % e)
            continue
        print("   引擎 %.1fs  最佳=%s  领先=%.1f目" % (k["dt"], k["best"], k["lead"]))
        print("   候选: %s" % "  ".join("%s(%.1f)" % (m, l) for m, l in k["cands"]))
        print("   PV: %s" % " ".join(k["pv"]))
        # 可信度判读
        warn = []
        if abs(k["lead"]) > 40:
            warn.append("目差 %.0f 极大 → 疑似棋形识别有误（多子/少子/黑白颠倒）" % k["lead"])
        if not k["cands"]:
            warn.append("引擎无候选手 → 局面可能非法")
        if warn:
            for w in warn:
                print("   ⚠ %s" % w)
        else:
            print("   ✓ 局面被引擎正常接受")


if __name__ == "__main__":
    main()
