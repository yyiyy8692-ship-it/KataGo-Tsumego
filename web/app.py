"""死活题拍照批改 App — Flask 后端。
流程：拍照上传(+选题型) → 棋形画回确认(硬性门禁) → 孩子答案录入/确认 → 批改报告。
"""
import json
import os
import sys
import uuid
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, jsonify, render_template, request, send_file, abort

import config
from recognition.detect import recognize
from recognition.render import board_svg
from grader.grader import grade, hints_for, PROBLEM_TYPES

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS = os.path.join(BASE, "uploads")
os.makedirs(UPLOADS, exist_ok=True)

app = Flask(__name__)


def _session_path(sid):
    return os.path.join(UPLOADS, sid, "session.json")


def _load(sid):
    p = _session_path(sid)
    if not os.path.exists(p):
        abort(404)
    with open(p) as f:
        return json.load(f)


def _save(sid, data):
    with open(_session_path(sid), "w") as f:
        json.dump(data, f, ensure_ascii=False)


@app.route("/")
def index():
    return render_template("index.html", ptypes=PROBLEM_TYPES)


@app.route("/api/upload", methods=["POST"])
def upload():
    photo = request.files.get("photo")
    ptype = request.form.get("ptype")
    if not photo or ptype not in PROBLEM_TYPES:
        return jsonify({"error": "缺少照片或题型（题型必选：黑先杀/黑先活/白先杀/白先活）"}), 400
    sid = uuid.uuid4().hex[:10]
    d = os.path.join(UPLOADS, sid)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "photo.jpg")
    photo.save(path)
    try:
        rec = recognize(path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    sess = {"sid": sid, "ptype": ptype, "note": request.form.get("note", ""),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "photo": path, "rec": {k: v for k, v in rec.items() if k != "overlay"},
            "overlay": os.path.basename(rec["overlay"])}
    _save(sid, sess)
    return jsonify({"sid": sid, "rec": sess["rec"],
                    "overlay_url": f"/file/{sid}/{sess['overlay']}"})


@app.route("/file/<sid>/<name>")
def file(sid, name):
    if "/" in name or ".." in name:
        abort(400)
    return send_file(os.path.join(UPLOADS, sid, name))


@app.route("/api/confirm", methods=["POST"])
def confirm():
    """画回确认门禁：家长确认/修改后的棋形才进入批改。"""
    j = request.get_json(force=True)
    sess = _load(j["sid"])
    sess["confirmed"] = {"black": [tuple(p) for p in j["black"]],
                         "white": [tuple(p) for p in j["white"]]}
    _save(j["sid"], sess)
    return jsonify({"ok": True})


@app.route("/api/grade", methods=["POST"])
def do_grade():
    j = request.get_json(force=True)
    sess = _load(j["sid"])
    if "confirmed" not in sess:
        return jsonify({"error": "请先确认棋形（画回确认是硬性门禁）"}), 400
    rec = sess["rec"]
    kid = [(m["seq"], m["color"], m["x"], m["y"]) for m in j.get("kid_moves", [])]
    try:
        result = grade(rec["cols"], rec["rows"],
                       sess["confirmed"]["black"], sess["confirmed"]["white"],
                       sess["ptype"], kid, note=sess.get("note", ""))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    sess["kid_moves"] = j.get("kid_moves", [])
    sess["result"] = result
    if j.get("archive", True):  # 测试时可传 archive:false 免污染题库
        try:
            from grader.archive import archive
            sess["problem_id"] = archive(sess, result)
        except Exception as e:
            sess["archive_error"] = str(e)  # 归档失败不阻塞批改
    _save(j["sid"], sess)
    return jsonify({"result": result, "hints": hints_for(sess["ptype"]),
                    "problem_id": sess.get("problem_id")})


@app.route("/report/<sid>")
def report(sid):
    sess = _load(sid)
    if "result" not in sess:
        abort(404)
    rec, res = sess["rec"], sess["result"]
    cols, rows = rec["cols"], rec["rows"]
    black = [tuple(p) for p in sess["confirmed"]["black"]]
    white = [tuple(p) for p in sess["confirmed"]["white"]]
    q_svg = board_svg(cols, rows, black, white, caption="题目")
    # 正解图：正解首着 + PV 前 6 手
    from engine.katago import from_gtp
    to_play = PROBLEM_TYPES[sess["ptype"]][0]
    pl, sol_moves, sx, sy = to_play, [], None, None
    for i, m in enumerate(res["solution"]["pv"][:6]):
        if m == "pass":
            continue
        x, y = from_gtp(m)
        if 0 <= x < cols and 0 <= y < rows:
            sol_moves.append((i + 1, pl, x, y))
            if i == 0:
                sx, sy = x, y
        pl = "W" if pl == "B" else "B"
    sol_svg = board_svg(cols, rows, black, white, moves=sol_moves,
                        caption=f"正解：{res['solution']['move']}，{res['solution']['verdict']}")
    kid_svg = ""
    if sess.get("kid_moves"):
        km = [(m["seq"], m["color"], m["x"], m["y"]) for m in sess["kid_moves"]]
        kid_svg = board_svg(cols, rows, black, white, moves=km,
                            caption=f"孩子的变化（结局：{res.get('kid_verdict') or '未录'}）")
    return render_template("report.html", sess=sess, res=res,
                           q_svg=q_svg, sol_svg=sol_svg, kid_svg=kid_svg,
                           hints=hints_for(sess["ptype"]))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=False)
