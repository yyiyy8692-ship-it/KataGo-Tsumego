"""自动化测试：识别层（合成图全场景）+ API 流程（需服务已启动）。
用法：python tests/run_tests.py           # 只测识别层
     python tests/run_tests.py --api     # 识别层 + API 全流程
"""
import json
import os
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PY = sys.executable
FAILED = []


def check(name, cond, detail=""):
    print(("PASS" if cond else "FAIL"), name, detail if not cond else "")
    if not cond:
        FAILED.append(name)


def gen_fixtures():
    from make_synthetic import make_photo
    import cv2, numpy as np
    make_photo(7, 5, black=[(1,1),(2,1),(3,1),(1,3),(2,3)],
               white=[(1,2),(2,2),(3,2),(4,2),(3,3)],
               digits=[(1,3,0),(2,4,1),(3,2,0)], out="/tmp/t_p1.jpg", rot=2.0)
    make_photo(5, 5, black=[(1,1),(2,1),(3,1)], white=[(1,2),(2,2),(3,2),(2,3)],
               digits=[], out="/tmp/t_small.jpg", rot=-3.0)
    img = cv2.imread("/tmp/t_p1.jpg")
    h, w = img.shape[:2]
    k = 0.06
    M = cv2.getPerspectiveTransform(
        np.float32([[0,0],[w,0],[w,h],[0,h]]),
        np.float32([[w*k,h*k],[w*(1-k),h*k*0.5],[w*(1-k*0.3),h],[w*k*0.3,h]]))
    cv2.imwrite("/tmp/t_persp.jpg", cv2.warpPerspective(img, M, (w,h), borderValue=(245,242,235)))


def test_recognition():
    from recognition.detect import recognize
    eb = {(1,1),(2,1),(3,1),(1,3),(2,3)}
    ew = {(1,2),(2,2),(3,2),(4,2),(3,3)}

    r = recognize("/tmp/t_p1.jpg")
    check("识别-旋转2°-棋子", set(map(tuple,r["black"]))==eb and set(map(tuple,r["white"]))==ew)
    check("识别-旋转2°-编号", [d[0] for d in r["digits"]]==[1,2,3], str(r["digits"]))

    r = recognize("/tmp/t_small.jpg")
    check("识别-小棋盘5x5-棋子", set(map(tuple,r["black"]))=={(1,1),(2,1),(3,1)}
          and set(map(tuple,r["white"]))=={(1,2),(2,2),(3,2),(2,3)},
          f"B={r['black']} W={r['white']}")

    r = recognize("/tmp/t_persp.jpg")
    check("识别-透视-棋子", set(map(tuple,r["black"]))==eb and set(map(tuple,r["white"]))==ew,
          f"B={r['black']} W={r['white']}")
    seqs = {(d[1],d[2]): d[0] for d in r["digits"]}
    check("识别-透视-误读宁缺毋错", seqs.get((4,1)) is None, f"2被读成{seqs.get((4,1))}")
    check("识别-透视-其余编号", seqs.get((3,0))==1 and seqs.get((2,0))==3, str(r["digits"]))

    try:
        recognize("/tmp/synth_blank.jpg")
        check("识别-白纸报错", False, "未报错")
    except ValueError:
        check("识别-白纸报错", True)


def api(method, path, data=None, files=None):
    url = "http://127.0.0.1:5050" + path
    if files:
        import subprocess as sp
        cmd = ["curl", "-s", "-X", "POST"]
        for k, v in files.items():
            cmd += ["-F", f"{k}=@{v}"]
        for k, v in (data or {}).items():
            cmd += ["-F", f"{k}={v}"]
        out = sp.check_output(cmd + [url])
        return json.loads(out)
    req = urllib.request.Request(
        url, data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"}, method=method)
    return json.loads(urllib.request.urlopen(req).read())


def test_api():
    r = api("POST", "/api/upload", {"ptype": "B_kill"}, {"photo": "/tmp/t_p1.jpg"})
    check("API-上传", "sid" in r, str(r)[:120])
    sid = r["sid"]
    rec = r["rec"]
    r = api("POST", "/api/confirm", {"sid": sid, "black": rec["black"], "white": rec["white"]})
    check("API-确认棋形", r.get("ok"))
    r = api("POST", "/api/grade", {"sid": sid, "archive": False, "kid_moves": [
        {"seq":1,"color":"B","x":3,"y":0},{"seq":2,"color":"W","x":4,"y":1}]})
    check("API-批改", "result" in r and r["result"]["solution"]["move"], str(r)[:120])
    # 归档单独验证：直接调 archive 函数写入临时题库，不碰生产 problems.json
    import tempfile, shutil as _sh
    from grader import archive as arc
    tmp = tempfile.mktemp(suffix=".json")
    _sh.copy2(arc.PROBLEMS, tmp)
    arc.PROBLEMS = tmp
    sess = {"rec": rec, "ptype": "B_kill", "note": "",
            "confirmed": {"black": rec["black"], "white": rec["white"]},
            "photo": "/tmp/t_p1.jpg"}
    pid = arc.archive(sess, r["result"])
    entry = next(p for p in json.load(open(tmp)) if p["id"] == pid)
    check("API-题库条目完整", bool(entry["sol"] and entry["B"] and entry["explain"]), pid)
    check("API-归档开关", "problem_id" not in r or not r.get("problem_id"))
    req = urllib.request.urlopen(f"http://127.0.0.1:5050/report/{sid}")
    html = req.read().decode()
    check("API-讲题卡SVG", html.count("<svg") >= 2)
    r = api("POST", "/api/upload", {}, {"photo": "/tmp/t_p1.jpg"})
    check("API-缺题型报错", "error" in r)


if __name__ == "__main__":
    gen_fixtures()
    test_recognition()
    if "--api" in sys.argv:
        test_api()
    print(f"\n{'全部通过 ✅' if not FAILED else f'{len(FAILED)} 项失败: {FAILED}'}")
    sys.exit(1 if FAILED else 0)
