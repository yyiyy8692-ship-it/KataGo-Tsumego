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
from recognition.detect import recognize, split_boards
from recognition.render import board_svg
from grader.grader import grade, hints_for, PROBLEM_TYPES
from web import solve_service   # 答案模式服务层（异步任务与求解管线）

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
    """答案模式（默认）：拍图 / 相册 / PDF 三入口，直接出正解。"""
    return render_template("solve.html")


@app.route("/grade")
def grade_page():
    """批改模式（旧 session 流程）：要选题型、录孩子的下法。"""
    import json as _json
    return render_template("index.html", ptypes=PROBLEM_TYPES,
                           ptypes_json=_json.dumps(
                               {k: v[2] for k, v in PROBLEM_TYPES.items()}))


# ----------------------------------------------------------- 答案模式 API
@app.route("/api/solve", methods=["POST"])
def api_solve():
    """上传一张照片或一份 PDF，起一个后台任务。

    全部按**黑先**计算（老师给的题册默认如此），暂不开放走棋方选项——
    绝大部分题册印刷的就是黑先，加个开关只会多一处容易误操作的入口。
    """
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"error": "没有收到文件"}), 400
    opts = {
        "depth": int(request.form.get("depth", 5)),
        "visits": int(request.form.get("visits", 3000)),
        "tol": float(request.form.get("tol", 1.0)),
    }
    try:
        tid = solve_service.create_task(f, opts)
    except Exception as e:
        return jsonify({"error": "上传失败：%s" % e}), 400
    return jsonify({"tid": tid})


@app.route("/api/task/<tid>")
def api_task(tid):
    """任务状态。?after=N 只回传第 N 题之后的（前端增量接收，省流量）。"""
    try:
        after = int(request.args.get("after", 0))
    except ValueError:
        after = 0
    v = solve_service.task_view(tid, after)
    if v is None:
        return jsonify({"error": "任务不存在或已过期"}), 404
    return jsonify(v)


@app.route("/solve/<tid>")
def solve_page(tid):
    """结果页（可刷新、可分享链接）。"""
    if solve_service.task_view(tid) is None:
        abort(404)
    return render_template("solve.html", tid=tid)


def _board_ctx(sess, j):
    """单题会话返回 sess 本身；多题会话返回第 board 个棋盘子 dict（就地改，
    _save 后落盘）。"""
    if sess.get("multi"):
        try:
            i = int(j.get("board", 0) or 0)
        except (TypeError, ValueError):
            abort(400)
        boards = sess["boards"]
        if not (0 <= i < len(boards)):
            abort(400)
        return boards[i]
    return sess


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
        crops = split_boards(path)  # 一图多题切分；单棋盘返回 [原图]
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception:
        crops = [path]
    if len(crops) == 1:
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
    # 多题：逐题识别（识别失败的单题保留 error，前端标出但不阻塞其他题）
    boards = []
    for cp in crops:
        b = {"photo": cp}
        try:
            rec = recognize(cp)
            b["rec"] = {k: v for k, v in rec.items() if k != "overlay"}
            b["overlay"] = os.path.basename(rec["overlay"])
        except ValueError as e:
            b["error"] = str(e)
        boards.append(b)
    if not any("rec" in b for b in boards):
        return jsonify({"error": "未检测到棋盘网格，请重拍（正对题图、光线均匀、题图完整入镜）"}), 422
    sess = {"sid": sid, "ptype": ptype, "note": request.form.get("note", ""),
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "photo": path, "multi": True, "boards": boards}
    _save(sid, sess)
    return jsonify({
        "sid": sid, "multi": True, "ptype": ptype,
        "boards": [{"rec": b.get("rec"), "error": b.get("error"),
                    "thumb_url": f"/file/{sid}/{os.path.basename(b['photo'])}",
                    "overlay_url": (f"/file/{sid}/{b['overlay']}"
                                    if "overlay" in b else None)}
                   for b in boards]})


@app.route("/file/<sid>/<name>")
def file(sid, name):
    if "/" in name or ".." in name:
        abort(400)
    return send_file(os.path.join(UPLOADS, sid, name))


@app.route("/api/confirm", methods=["POST"])
def confirm():
    """画回确认门禁：家长确认/修改后的棋形才进入批改。
    多题会话需带 board 序号；ptype 可在此按题覆盖（默认整页统一）。"""
    j = request.get_json(force=True)
    sess = _load(j["sid"])
    ctx = _board_ctx(sess, j)
    if "rec" not in ctx:
        return jsonify({"error": "该题识别失败，请单独重拍这一题"}), 400
    ctx["confirmed"] = {"black": [tuple(p) for p in j["black"]],
                        "white": [tuple(p) for p in j["white"]]}
    if j.get("ptype") in PROBLEM_TYPES:
        ctx["ptype"] = j["ptype"]
    _save(j["sid"], sess)
    return jsonify({"ok": True})


@app.route("/api/grade", methods=["POST"])
def do_grade():
    j = request.get_json(force=True)
    sess = _load(j["sid"])
    ctx = _board_ctx(sess, j)
    if "confirmed" not in ctx:
        return jsonify({"error": "请先确认棋形（画回确认是硬性门禁）"}), 400
    rec = ctx["rec"]
    ptype = ctx.get("ptype", sess["ptype"])
    kid = [(m["seq"], m["color"], m["x"], m["y"]) for m in j.get("kid_moves", [])]
    try:
        result = grade(rec["cols"], rec["rows"],
                       ctx["confirmed"]["black"], ctx["confirmed"]["white"],
                       ptype, kid, note=sess.get("note", ""))
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    ctx["kid_moves"] = j.get("kid_moves", [])
    ctx["result"] = result
    if j.get("archive", True):  # 测试时可传 archive:false 免污染题库
        try:
            from grader.archive import archive
            # archive 需要 {rec, ptype, confirmed, note, photo} 形状的视图
            view = {"rec": rec, "ptype": ptype, "note": sess.get("note", ""),
                    "confirmed": ctx["confirmed"], "photo": ctx.get("photo", sess["photo"])}
            ctx["problem_id"] = archive(view, result)
        except Exception as e:
            ctx["archive_error"] = str(e)  # 归档失败不阻塞批改
    _save(j["sid"], sess)
    return jsonify({"result": result, "hints": hints_for(ptype),
                    "problem_id": ctx.get("problem_id")})


@app.route("/report/<sid>")
@app.route("/report/<sid>/<int:bi>")
def report(sid, bi=None):
    sess = _load(sid)
    ctx = sess
    if sess.get("multi"):
        boards = sess["boards"]
        if bi is None:  # 未指定时给第一道已批改的题
            ctx = next((b for b in boards if "result" in b), None)
        elif 0 <= bi < len(boards):
            ctx = boards[bi]
        if ctx is None or "result" not in ctx:
            abort(404)
    if "result" not in ctx:
        abort(404)
    rec, res = ctx["rec"], ctx["result"]
    ptype = ctx.get("ptype", sess["ptype"])
    cols, rows = rec["cols"], rec["rows"]
    black = [tuple(p) for p in ctx["confirmed"]["black"]]
    white = [tuple(p) for p in ctx["confirmed"]["white"]]
    q_svg = board_svg(cols, rows, black, white, caption="题目")
    # 正解图：正解首着 + PV 前 6 手
    from engine.katago import from_gtp
    to_play = PROBLEM_TYPES[ptype][0]
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
    if ctx.get("kid_moves"):
        km = [(m["seq"], m["color"], m["x"], m["y"]) for m in ctx["kid_moves"]]
        kid_svg = board_svg(cols, rows, black, white, moves=km,
                            caption=f"孩子的变化（结局：{res.get('kid_verdict') or '未录'}）")
    view = {"sid": sid, "created": sess.get("created", ""),
            "problem_id": ctx.get("problem_id"),
            "board": (bi + 1) if sess.get("multi") and bi is not None else None,
            "board_count": len(sess["boards"]) if sess.get("multi") else None}
    return render_template("report.html", sess=view, res=res,
                           q_svg=q_svg, sol_svg=sol_svg, kid_svg=kid_svg,
                           hints=hints_for(ptype))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=config.PORT, debug=False)
