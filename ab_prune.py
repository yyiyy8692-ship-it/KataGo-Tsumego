"""对照实验：网格剪枝开关对识别结果的影响（针对 page1 整页的全局题）。"""
import sys

import recognition.detect as D

PHOTOS = ["photos/page1_b0.png", "photos/page1_b1.png", "photos/page1_b2.png"]


def run(tag):
    for p in PHOTOS:
        try:
            g = D.prep_grid(p)
            xs, ys = g["xs"], g["ys"]
            print("%-8s %s  %d列 x %d行" % (tag, p.split("/")[-1],
                                            len(xs), len(ys)))
        except Exception as e:
            print("%-8s %s  ERROR %s" % (tag, p.split("/")[-1], e))


run("剪枝开")
# 关掉剪枝（模拟提交版 954def9 的行为）
_orig = D._prune_low_coverage_lines
D._prune_low_coverage_lines = lambda gray, xs, ys, min_cov=0.30: (xs, ys)
run("剪枝关")
D._prune_low_coverage_lines = _orig
