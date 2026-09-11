"""对整页切分出的每道题生成「正解 + 3~5 步变化图」（轻量版）。

与 page_review.py 的区别：page_review 出的是**识别对照图**（原图 vs 电子棋盘，
用来核对识别对不对）；这里出的是**讲题图**（正解 + 双方最优应对的变化序列 +
目差），不需要录入孩子的下法。

用法：
    python solve_page.py photos/paper_p1_300.png solution_out
    python solve_page.py photos/paper_p1_300.png solution_out 5 3000
                                                  ↑depth ↑visits
"""
import os
import sys

sys.path.insert(0, ".")

from recognition import detect as D
from recognition import corner
import fullboard as F
import solver

# 题名按切分顺序（PDF 页面文字顺序）
DEFAULT_NAMES = ["全局题1", "全局题2", "全局题3",
                 "手筋题1", "手筋题2", "手筋题3"]


def recognize(path):
    """返回 (cols, rows, black, white, corner)。整盘优先，失败回退局部管线。"""
    try:
        r = F.recognize_full_board(path)
        if r and r.get("cols", 0) >= 15:
            return (r["cols"], r["rows"], list(r["black"]), list(r["white"]),
                    "FULL")
    except Exception:
        pass
    r = D.recognize(path)
    c = corner.locate(path).get("corner") or "BR"
    return r["cols"], r["rows"], list(r["black"]), list(r["white"]), c


def write_html(outdir, items, title="正解与变化图"):
    """把各题的变化图 SVG 内联成一页 HTML。

    为什么额外出 HTML：SVG 单文件在部分预览器里打不开，内联进 HTML 后可直接
    在浏览器里看、可缩放、可整页打印（讲题时打印给孩子更实用）。
    """
    css = ("body{font-family:PingFang SC,-apple-system,sans-serif;"
           "max-width:1180px;margin:0 auto;padding:24px;background:#F7F8FA;}"
           "h1{font-size:22px;font-weight:500;margin:0 0 4px;}"
           ".lead{color:#5F5E5A;font-size:14px;margin:0 0 24px;}"
           "section{background:#fff;border-radius:12px;padding:20px;"
           "margin-bottom:20px;}"
           "h2{font-size:17px;font-weight:500;margin:0 0 12px;}"
           "svg{width:100%;height:auto;}@media print{body{background:#fff}}")
    out = os.path.join(outdir, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html lang=\"zh\"><head><meta charset=\"utf-8\">")
        f.write("<title>%s</title><style>%s</style></head><body>" % (title, css))
        f.write("<h1>%s</h1>" % html_escape(title))
        f.write("<p class=\"lead\">共 %d 张图 ｜ 全部黑先 ｜ "
                "一张棋盘标出完整变化，序号写在棋子正中央 ｜ "
                "目差均为黑方视角 ｜ 全局题只给第 1 手，双解题每个正解各一张</p>"
                % len(items))
        for name, svg in items:
            f.write("<section><h2>%s</h2>%s</section>"
                    % (html_escape(name), svg))
        f.write("</body></html>")
    return out


def html_escape(s):
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def main(src, outdir, depth=5, visits=3000, names=None, only=None, tol=1.0):
    """only: 只做其中几题（0-based 下标列表），调试用。

    tol: 两个候选目差相差 <= tol 目就算等价正解（双解一起列出来）。
    """
    os.makedirs(outdir, exist_ok=True)
    names = names or DEFAULT_NAMES
    stem = os.path.basename(src).rsplit(".", 1)[0]
    parts = ["photos/%s_b%d.png" % (stem, i) for i in range(len(names))]
    parts = [p for p in parts if os.path.exists(p)]
    if not parts:
        parts = D.split_boards(src)
    print("共 %d 题，depth=%d visits=%d" % (len(parts), depth, visits))
    outs, items = [], []
    for i, p in enumerate(parts):
        if only is not None and i not in only:
            continue
        name = names[i] if i < len(names) else "第%d题" % (i + 1)
        cols, rows, black, white, c = recognize(p)
        print("--- 第%d题 %s  %dx%d 黑%d 白%d"
              % (i + 1, name, cols, rows, len(black), len(white)))
        # 全局题（19 路实战局面）只给第 1 手；局部题列出全部等价正解
        is_global = cols >= 15 or c == "FULL"
        if is_global:
            print("    题型：全局实战题 → 只出第 1 手")
            multi = {"solutions": [solver.solve(
                cols, rows, black, white, to_play="B", depth=1,
                visits=visits, kind="global")], "alerts": []}
        else:
            multi = solver.solve_multi(cols, rows, black, white, to_play="B",
                                       depth=depth, visits=visits, tol=tol)
            print("    正解 %d 个：%s"
                  % (multi["n_solutions"],
                     " / ".join("%s(%.1f)" % (r["best"], r["lead"])
                                for r in multi["solutions"])))
        for a in multi["alerts"]:
            print("    注意 %s" % a)

        sols = multi["solutions"]
        for k, res in enumerate(sols):
            for s in res["steps"]:
                print("    第%d手 %s %-4s 领先%+.1f 增益%+.1f%s [%s vis=%d]"
                      % (s["seq"], s["color"], s["move"], s["lead"], s["gain"],
                         " 提%d子" % s["captured"] if s["captured"] else "",
                         s["criterion"], s["visits"]))
            print("    走完 %d 手后黑领先 %+.1f 目（起手 %+.1f）"
                  % (len(res["steps"]), res["final_lead"], res["lead0"]))
            for a in res["alerts"]:
                print("    注意 %s" % a)
            tag = ""
            if is_global:
                tag = " · 最大一手"
            elif len(sols) > 1:
                tag = " · 正解%d/%d（%s）" % (k + 1, len(sols), res["best"])
            title = "第%d题 %s%s" % (i + 1, name, tag)
            svg = solver.variation_single(title, cols, rows, c, black, white,
                                          res)
            # 标题里的 "正解1/2" 含斜杠，不能进文件名
            out = os.path.join(
                outdir, "%s.svg" % title.replace(" ", "_").replace("/", "-"))
            with open(out, "w", encoding="utf-8") as f:
                f.write(svg)
            outs.append(out)
            items.append((title, svg))
            print("    -> %s" % out)
    print("    -> %s" % write_html(outdir, items))
    return outs


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "photos/paper_p1_300.png"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "solution_out"
    depth = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    visits = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    only = ([int(v) for v in sys.argv[5].split(",")]
            if len(sys.argv) > 5 else None)
    main(src, outdir, depth, visits, only=only)
