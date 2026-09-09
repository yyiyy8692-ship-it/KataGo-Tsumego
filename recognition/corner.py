"""角（corner）定位：判定题图局部的实墙在哪两条边，也就是墙角是哪个角。

为什么必须有这一层
------------------
印刷死活题是 7x9 一类的局部块，恰好两条边画成粗实线（真墙＝棋盘边），另两条是普通
网格线（开放＝棋盘继续延伸）。哪个角是墙角决定了棋形要不要翻转——7x9 不是正方形，
四种角对应原位 / 左右镜像 / 上下镜像 / 180° 旋转，是四道完全不同的题。判错不会报错，
只会静默给出一个"看着合理但错误"的答案，所以这里必须输出置信度而不是硬判。

注意一个反直觉的点：**摆到 19 路的哪个角不影响计算结果**（棋盘旋转镜像对称，"棋盘边
＋白墙"构成的边界条件也对称）。关键只有"哪两条边是墙"。

三条证据
--------
1 线宽比   墙线 14-15px / 网格线 9-10px，取比值。这是印刷规则本身，最强的一条。
2 出头     墙线是否延伸过网格交点（真墙是边框画到头，网格线止于交点）。
5 同册先验 一本书排版规则固定，人工确认过的题锁定后续题目的先验。
（"外侧留白宽度""题号文字位置"两条 2026-09-09 经用户确认删除：依赖版面、不稳定。）

用法
----
    from recognition.corner import locate, record_confirmation
    r = locate("photos/q3.jpg")
    r["corner"]        # "BR"
    r["confidence"]    # 0.42
    r["need_confirm"]  # 置信度不足，应走人工确认或引擎交叉验证
    r["scores"]        # {"TL": .., "TR": .., "BL": .., "BR": ..}

    locate(path, force="BL")              # 人工指定，跳过分类
    record_confirmation("default", "BR")  # 用户确认/纠正后写回同册先验

线位来源：复用 detect.prep_grid 的 xs/ys（透视矫正后、逆透视 warp 之前）。
warp 之后最外四条线被拉到图像边界上，线宽和出头都量不到了，所以不能在那一步做。
"""
import json
import os

import numpy as np

from . import detect as D

CORNERS = ("TL", "TR", "BL", "BR")
WALL_EDGES = {"TL": ("L", "T"), "TR": ("R", "T"),
              "BL": ("L", "B"), "BR": ("R", "B")}
EDGES = ("L", "R", "T", "B")

PROFILE_PATH = os.path.expanduser("~/.workbuddy/tsume_corner_profiles.json")

# 置信度门槛：低于此值不能静默采信，要人工确认或跑引擎交叉验证
# 角判定置信门槛。实测下限：测量被图内暗结构污染时（铅笔手写压内部参考线）
# 置信会掉到 0.2 附近并可能误判（q4 实测 0.213 判错 BL），0.15 拦不住它；
# 正常样本 0.18~0.51（p2 最低 0.182）——与 0.25 之间有重叠，门槛只能牺牲
# p2 这类"低置信但判对"的样本让它多走一次人工确认，不能放过误判。
CONFIRM_THRESHOLD = 0.25


# ---------------------------------------------------------------- 原始测量

_width_at = D._line_width_at   # 线宽测量与 detect 共用一份实现


def _sample_widths(gray, pos, axis, lo, hi, n_sample=140, margin=18):
    """沿一条线采 n_sample 个点量宽度，返回中位数与有效样本率。

    axis="v" 线是竖直的（法向水平）；axis="h" 线是水平的（法向竖直）。
    宽度 >25px 的样本是棋子或文字，直接丢掉——线只有 9-15px。
    """
    if hi - lo < 2 * margin + 10:
        margin = 0
    ts = np.linspace(lo + margin, hi - margin, n_sample)
    ws = []
    for t in ts:
        w = (_width_at(gray, pos, t, 1, 0) if axis == "v"
             else _width_at(gray, t, pos, 0, 1))
        if w is not None and 2 <= w <= 25:
            ws.append(w)
    if not ws:
        return None, 0.0
    return float(np.median(ws)), len(ws) / len(ts)


def _continuation(gray, pos, axis, end, outward, max_scan=30):
    """从线的端点 end 沿自身方向朝外扫，返回线还延续了多少像素（出头长度）。

    不能用"整条线的连通跨度"来算：白子压在线上会把线打断，连通段提前结束，
    实测算出 -1032 这种荒谬值。改成从端点向外逐像素探，允许 5px 中断。
    """
    run, miss = 0, 0
    for k in range(1, max_scan + 1):
        t = end + outward * k
        w = (_width_at(gray, pos, t, 1, 0) if axis == "v"
             else _width_at(gray, t, pos, 0, 1))
        if w is not None and 2 <= w <= 25:
            run, miss = k, 0
        else:
            miss += 1
            if miss >= 5:
                break
    return run


def _edge_overshoot(gray, pos, axis, lo, hi):
    """一条线两端各自的出头长度，取两端之和。"""
    a = _continuation(gray, pos, axis, lo, -1)
    b = _continuation(gray, pos, axis, hi, +1)
    return a + b, a, b


def measure(photo_path):
    """量四条边与内部线的原始数值。返回 dict，全部是未归一化的原始量。"""
    P = D.prep_grid(photo_path)
    gray, xs, ys = P["gray"], P["xs"], P["ys"]
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])

    # 内部线基准：竖线用竖直方向的中位，横线另算（印刷横竖线宽可能不同）
    ref = {}
    for tag, lines, axis in (("v", xs[1:-1], "v"), ("h", ys[1:-1], "h")):
        ws = []
        for p in lines:
            lo, hi = ((y0, y1) if axis == "v" else (x0, x1))
            w, _ = _sample_widths(gray, float(p), axis, lo, hi, n_sample=90)
            if w:
                ws.append(w)
        ref[tag] = float(np.median(ws)) if ws else None

    raw = {}
    spec = {"L": (x0, "v", y0, y1), "R": (x1, "v", y0, y1),
            "T": (y0, "h", x0, x1), "B": (y1, "h", x0, x1)}
    for tag, (pos, axis, lo, hi) in spec.items():
        w, rate = _sample_widths(gray, pos, axis, lo, hi)
        tot, oa, ob = _edge_overshoot(gray, pos, axis, lo, hi)
        r = ref["v"] if axis == "v" else ref["h"]
        raw[tag] = {
            "width": w,
            "ratio": (w / r) if (w and r) else None,
            "overshoot": tot,
            "overshoot_a": oa,
            "overshoot_b": ob,
            "sample_rate": round(rate, 2),
        }
    return {"ref": ref, "edges": raw, "cols": len(xs), "rows": len(ys),
            "xs": [float(v) for v in xs], "ys": [float(v) for v in ys]}


# ---------------------------------------------------------------- 打分

def _ramp(v, lo, hi):
    """把原始量线性压到 [0,1]。"""
    if v is None:
        return 0.5
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def _edge_scores(m):
    """每条边的"这是墙"分数。线宽比用 1.10→1.45 斜坡，出头用 0→12px 斜坡。"""
    out = {}
    for tag in EDGES:
        e = m["edges"][tag]
        out[tag] = {
            "thick": _ramp(e["ratio"], 1.10, 1.45),
            "over": _ramp(e["overshoot"], 3.0, 15.0),
        }
    return out


def _load_profiles():
    if not os.path.exists(PROFILE_PATH):
        return {}
    try:
        with open(PROFILE_PATH) as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _prior(profile):
    """同册先验：确认次数越多权重越高，最多占到 0.25。"""
    data = _load_profiles().get(profile, {})
    counts = {c: int(data.get("counts", {}).get(c, 0)) for c in CORNERS}
    n = sum(counts.values())
    if n == 0:
        return {c: 0.0 for c in CORNERS}, 0.0
    return {c: counts[c] / n for c in CORNERS}, 0.25 * n / (n + 2.0)


def record_confirmation(profile, corner):
    """用户确认/纠正后调用，写回同册先验。corner ∈ CORNERS。"""
    if corner not in CORNERS:
        raise ValueError("corner 必须是 %s 之一" % (CORNERS,))
    data = _load_profiles()
    rec = data.setdefault(profile, {"counts": {}, "history": []})
    counts = rec.setdefault("counts", {})
    counts[corner] = int(counts.get(corner, 0)) + 1
    rec["history"].append(corner)
    rec["history"] = rec["history"][-50:]
    os.makedirs(os.path.dirname(PROFILE_PATH), exist_ok=True)
    with open(PROFILE_PATH, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    return dict(counts)


def locate(photo_path, profile="default", force=None):
    """判定墙角。返回 corner / confidence / scores / 原始测量值。

    force 传入 "TL"/"TR"/"BL"/"BR" 时跳过分类（人工兜底用）。
    """
    m = measure(photo_path)
    es = _edge_scores(m)
    prior, w_prior = _prior(profile)
    w_rest = 1.0 - w_prior
    # 出头权重刻意压到 0.15：2026-09-09 实测它在四题样本上不具区分度
    # （p1 上甚至指向反方向，q4 有三条边顶到扫描上限），只能当平局裁决，
    # 不能让它翻掉线宽比的结论。若后续换题册仍无区分度，应直接摘掉这条证据。
    w_thick, w_over = w_rest * 0.85, w_rest * 0.15

    scores = {}
    for c in CORNERS:
        wl = WALL_EDGES[c]
        terms = []
        for e in EDGES:
            s = es[e]["thick"] if e in wl else 1.0 - es[e]["thick"]
            terms.append(s * w_thick)
            o = es[e]["over"] if e in wl else 1.0 - es[e]["over"]
            terms.append(o * w_over)
        scores[c] = sum(terms) / 4.0 + w_prior * prior[c]

    order = sorted(CORNERS, key=lambda c: -scores[c])
    top1, top2 = order[0], order[1]
    conf = scores[top1] - scores[top2]

    if force:
        if force not in CORNERS:
            raise ValueError("force 必须是 %s 之一" % (CORNERS,))
        return {"corner": force, "confidence": 1.0, "forced": True,
                "need_confirm": False,
                "scores": scores, "measure": m, "edge_scores": es,
                "weights": {"thick": w_thick, "over": w_over, "prior": w_prior}}

    return {"corner": top1, "confidence": round(float(conf), 3),
            "forced": False, "scores": {c: round(float(scores[c]), 3)
                                        for c in CORNERS},
            "need_confirm": bool(conf < CONFIRM_THRESHOLD),
            "measure": m, "edge_scores": {k: {kk: round(vv, 3)
                                              for kk, vv in v.items()}
                                          for k, v in es.items()},
            "weights": {"thick": round(w_thick, 3), "over": round(w_over, 3),
                        "prior": round(w_prior, 3)}}
