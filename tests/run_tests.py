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
    np.random.seed(42)  # 夹具噪声固定，避免偶发抖动造成假失败
    make_photo(7, 5, black=[(1,1),(2,1),(3,1),(1,3),(2,3)],
               white=[(1,2),(2,2),(3,2),(4,2),(3,3)],
               digits=[(1,3,0),(2,4,1),(3,2,0)], out="/tmp/t_p1.jpg", rot=2.0)
    make_photo(5, 5, black=[(1,1),(2,1),(3,1)], white=[(1,2),(2,2),(3,2),(2,3)],
               digits=[], out="/tmp/t_small.jpg", rot=-3.0)
    # 线穿白子印刷风格 + 6° 侧拍梯形畸变（横线水平竖线斜）
    # ——2026-09-08 真实书页照片踩出的两个坑，固化成回归
    make_photo(7, 9, black=[(4,6),(4,7),(5,4),(5,5)],
               white=[(4,2),(5,3),(4,4),(4,5),(3,6),(1,7),(3,7)],
               digits=[], out="/tmp/t_linethrough.jpg", rot=0,
               line_through=True)
    img = cv2.imread("/tmp/t_linethrough.jpg")
    hh, ww = img.shape[:2]
    M = np.float32([[1, np.tan(np.radians(6.0)), 0], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (int(ww + hh * np.tan(np.radians(6.0))), hh),
                         borderValue=(245, 242, 235))
    cv2.imwrite("/tmp/t_linethrough.jpg", img)
    img = cv2.imread("/tmp/t_p1.jpg")
    h, w = img.shape[:2]
    k = 0.06
    M = cv2.getPerspectiveTransform(
        np.float32([[0,0],[w,0],[w,h],[0,h]]),
        np.float32([[w*k,h*k],[w*(1-k),h*k*0.5],[w*(1-k*0.3),h],[w*k*0.3,h]]))
    cv2.imwrite("/tmp/t_persp.jpg", cv2.warpPerspective(img, M, (w,h), borderValue=(245,242,235)))
    # 一页两题（模拟整页作业拍照）
    a = cv2.imread("/tmp/t_p1.jpg")
    b = cv2.imread("/tmp/t_small.jpg")
    gap = 60
    page = np.full((max(a.shape[0], b.shape[0]), a.shape[1] + gap + b.shape[1], 3),
                   (248, 246, 240), np.uint8)
    page[:a.shape[0], :a.shape[1]] = a
    page[:b.shape[0], a.shape[1] + gap:] = b
    cv2.imwrite("/tmp/t_multi.jpg", page)


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
    # 宁缺毋错：透视剪切的 2 允许标 ? 或读对（2），绝不允许误读（曾误读成 1）
    check("识别-透视-误读宁缺毋错", seqs.get((4,1)) in (None, 2), f"2被读成{seqs.get((4,1))}")
    check("识别-透视-其余编号", seqs.get((3,0))==1 and seqs.get((2,0))==3, str(r["digits"]))

    # 线穿白子 + 梯形畸变：真实书页场景回归（白子描边空心、线从子身穿过的
    # 印刷风格曾让白子全灭；6° 竖线倾斜曾让间距估计崩溃）
    r = recognize("/tmp/t_linethrough.jpg")
    check("识别-线穿白子梯形-棋子",
          set(map(tuple,r["black"]))=={(4,6),(4,7),(5,4),(5,5)}
          and set(map(tuple,r["white"]))=={(4,2),(5,3),(4,4),(4,5),(3,6),(1,7),(3,7)},
          f"B={r['black']} W={r['white']} grid={r['cols']}x{r['rows']}")

    try:
        recognize("/tmp/synth_blank.jpg")
        check("识别-白纸报错", False, "未报错")
    except ValueError:
        check("识别-白纸报错", True)

    # 一图多题：切分 + 逐题识别
    from recognition.detect import split_boards
    paths = split_boards("/tmp/t_multi.jpg")
    check("多题-切分出2题", len(paths) == 2, f"切出 {len(paths)}")
    if len(paths) == 2:
        r1 = recognize(paths[0])
        check("多题-第1题棋子", set(map(tuple, r1["black"])) == eb
              and set(map(tuple, r1["white"])) == ew,
              f"B={r1['black']} W={r1['white']}")
        check("多题-第1题编号零误读",
              all(d[0] in (None, 1, 2, 3) for d in r1["digits"]), str(r1["digits"]))
        r2 = recognize(paths[1])
        check("多题-第2题棋子", set(map(tuple, r2["black"])) == {(1,1),(2,1),(3,1)}
              and set(map(tuple, r2["white"])) == {(1,2),(2,2),(3,2),(2,3)},
              f"B={r2['black']} W={r2['white']}")
    # 单题照片切分必须原样返回（零回归）
    check("多题-单题不受影响", split_boards("/tmp/t_p1.jpg") == ["/tmp/t_p1.jpg"])


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
    # 一图多题 API 全流程
    r = api("POST", "/api/upload", {"ptype": "B_kill"}, {"photo": "/tmp/t_multi.jpg"})
    check("API-多题上传", r.get("multi") and len(r["boards"]) == 2, str(r)[:150])
    if r.get("multi") and len(r["boards"]) == 2:
        msid = r["sid"]
        b0, b1 = r["boards"]
        check("API-多题缩略图", bool(b0["thumb_url"] and b1["thumb_url"]))
        check("API-多题逐题识别", b0["rec"]["cols"] == 7 and b1["rec"]["cols"] == 5,
              f"b0={b0['rec']['cols']} b1={b1['rec']['cols']}")
        r = api("POST", "/api/confirm", {"sid": msid, "board": 1,
                                         "black": b1["rec"]["black"],
                                         "white": b1["rec"]["white"],
                                         "ptype": "W_kill"})
        check("API-多题确认", r.get("ok"), str(r)[:120])
        r = api("POST", "/api/grade", {"sid": msid, "board": 1, "archive": False,
                                       "kid_moves": []})
        check("API-多题批改", "result" in r and r["result"]["solution"]["move"], str(r)[:120])
        req = urllib.request.urlopen(f"http://127.0.0.1:5050/report/{msid}/1")
        html = req.read().decode()
        check("API-多题讲题卡", html.count("<svg") >= 2 and "第 2 / 2 题" in html)


if __name__ == "__main__":
    gen_fixtures()
    test_recognition()
    if "--api" in sys.argv:
        test_api()
    print(f"\n{'全部通过 ✅' if not FAILED else f'{len(FAILED)} 项失败: {FAILED}'}")
    sys.exit(1 if FAILED else 0)
