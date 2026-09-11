"""只重渲染：从 <outdir>/cache/*.json 重建全部 SVG 与 index.html。

为什么要它：求解一页 6 题要跑 4~5 分钟 KataGo，但调字号、间距、颜色这类
纯排版改动根本不依赖引擎。solve_page.py 现在会把每题的解算结果（含棋形与
各手目差）落盘到 cache/，本脚本读回来直接重画，秒级完成。

用法：
    python rerender.py solution_out
"""
import glob
import json
import os
import sys

sys.path.insert(0, ".")

import solve_page as S


def main(outdir="solution_out"):
    files = glob.glob(os.path.join(outdir, "cache", "b*.json"))
    if not files:
        raise SystemExit("没找到 %s/cache/*.json —— 先跑一次 solve_page.py" % outdir)
    files.sort(key=lambda p: int(os.path.basename(p)[1:-5]))
    items, n = [], 0
    for p in files:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        outs, its = S.render_item(outdir, d["idx"], d["name"], d["cols"],
                                  d["rows"], d["corner"], d["black"],
                                  d["white"], d["multi"], d["is_global"])
        for o in outs:
            print("-> %s" % o)
        items += its
        n += 1
    print("重渲染 %d 题 / %d 张图" % (n, len(items)))
    print("-> %s" % S.write_html(outdir, items))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "solution_out")
