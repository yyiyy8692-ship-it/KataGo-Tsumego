"""批改结果自动归档进题库 ~/go-review/data/problems.json。

小棋盘坐标 -> 19 路坐标：锚定右上角（列从 T 往左排，行从 19 往下排），
与题库 H 系列（作业实拍题）的坐标习惯一致。
归档前自动备份 problems.json.bak；写后校验 JSON 可解析。
"""
import json
import os
import shutil
from datetime import datetime

import config
from engine.katago import from_gtp

GC19 = "ABCDEFGHJKLMNOPQRST"
PROBLEMS = config.PROBLEMS
PHOTOS = config.PHOTOS

PTYPE_META = {
    "B_kill": ("B", "黑先，净杀白棋", "杀棋"),
    "B_live": ("B", "黑先，做活", "做活"),
    "W_kill": ("W", "白先，净杀黑棋", "杀棋"),
    "W_live": ("W", "白先，做活", "做活"),
}


def _to19(x, y, cols):
    return GC19[18 - (cols - 1) + x] + str(19 - y)


def archive(sess, result):
    """sess: 会话 dict；result: grade() 返回值。返回新条目 id。"""
    rec = sess["rec"]
    cols = rec["cols"]
    to_play, goal, cat = PTYPE_META[sess["ptype"]]
    with open(PROBLEMS) as f:
        problems = json.load(f)

    today = datetime.now().strftime("%Y%m%d")
    seq = sum(1 for p in problems if p["id"].startswith(f"A{today}")) + 1
    pid = f"A{today}-{seq}"

    black = [_to19(x, y, cols) for x, y in sess["confirmed"]["black"]]
    white = [_to19(x, y, cols) for x, y in sess["confirmed"]["white"]]
    gx, gy = from_gtp(result["solution"]["move"])
    sol19 = _to19(gx, gy, cols)

    parts = [f"引擎正解 {sol19}（{result['solution']['verdict']}）。"]
    if result.get("kid_report"):
        marks = []
        for m in result["kid_report"]:
            tag = "✓" if m["is_best"] else (f"第{m['rank']}选" if m["rank"] else f"✗应为{m['engine_best']}")
            kx, ky = from_gtp(m["move"])
            marks.append(f"{m['seq']}@{_to19(kx, ky, cols)}{tag}")
        parts.append("孩子：" + "、".join(marks) + f"，结局{result.get('kid_verdict')}。")
    if result.get("alerts"):
        parts.append("⚠️ " + "；".join(result["alerts"]))
    if sess.get("note"):
        parts.append(f"题面：{sess['note']}")

    entry = {
        "id": pid,
        "name": f"App批改 · {goal.split('，')[-1]} · {today}",
        "cat": cat, "level": None,
        "toPlay": to_play, "goal": goal,
        "B": black, "W": white, "sol": sol19,
        "explain": "".join(parts),
        "photo": "",
    }
    # 照片归档
    if sess.get("photo") and os.path.exists(sess["photo"]):
        os.makedirs(PHOTOS, exist_ok=True)
        dst = os.path.join(PHOTOS, f"{pid}.jpg")
        shutil.copy2(sess["photo"], dst)
        entry["photo"] = dst

    shutil.copy2(PROBLEMS, PROBLEMS + ".bak")
    problems.append(entry)
    with open(PROBLEMS, "w") as f:
        json.dump(problems, f, ensure_ascii=False, indent=1)
    json.load(open(PROBLEMS))  # 写后校验
    return pid
