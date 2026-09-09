"""消融实验：同一进程内 重新拼图 → 切分 → 识别，四种配置各跑一遍。

配置：全开 / 只关剪枝 / 只关门 / 都关。对比目标 = 人工核定 stones json。
"""
import json
import os
import sys

import recognition.detect as D
from recognition import corner as C

sys.path.insert(0, "/Users/yangyang/WorkBuddy/2026-09-08-18-38-57")
os.chdir("/Users/yangyang/tsume-app")

import importlib.util
spec = importlib.util.spec_from_file_location(
    "tms", "/Users/yangyang/WorkBuddy/2026-09-08-18-38-57/test_multi_split.py")
tms = importlib.util.module_from_spec(spec)
# 不执行 __main__ 块：只借 block_crop/compose/compose_grid/check
spec.loader.exec_module(tms) if False else None
# 手动加载模块内函数（exec 会跑 __main__，改为直接 exec 源码去掉尾部）
src = open("/Users/yangyang/WorkBuddy/2026-09-08-18-38-57/test_multi_split.py").read()
src = src.split('if __name__ == "__main__":')[0]
exec(compile(src, "tms", "exec"), tms.__dict__)

PH = tms.PH
orig_prune = D._prune_low_coverage_lines
orig_alias = D._aliased_lattice
NO_PRUNE = lambda gray, xs, ys, min_cov=0.15: (xs, ys)
NO_GATE = lambda *a, **k: False

CONFIGS = [
    ("全开", orig_prune, orig_alias),
    ("无剪枝", NO_PRUNE, orig_alias),
    ("无门", orig_prune, NO_GATE),
    ("都关", NO_PRUNE, NO_GATE),
]

for tag, prune, gate in CONFIGS:
    D._prune_low_coverage_lines = prune
    D._aliased_lattice = gate
    print("=== %s ===" % tag)
    parts = D.split_boards(tms.compose(["q3", "p2", "p1"],
                                       os.path.join(PH, "ab_multi_3.jpg")))
    tms.check(parts, ["q3", "p2", "p1"], "3题横排")
    parts = D.split_boards(tms.compose_grid(
        [["p1", "p2"], ["q3", "q4"]], os.path.join(PH, "ab_multi_2x2.jpg")))
    tms.check(parts, ["p1", "p2", "q3", "q4"], "2x2版面")

D._prune_low_coverage_lines = orig_prune
D._aliased_lattice = orig_alias
