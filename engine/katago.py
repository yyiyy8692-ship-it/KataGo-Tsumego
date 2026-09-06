"""KataGo 常驻 analysis 引擎封装。

协议坑（全部踩过，勿改）：
- initialStones 查询也必须带 "moves":[]
- 出错响应没有 turnNumber 字段——读取循环若只等 turnNumber 会永久挂起，必须显式 raise
- ownership 需要 "includeOwnership":true，且在响应顶层
- forced moves 一律用 gtp(x,y) 从坐标元组生成，禁止手拼 GTP 字符串
  （手拼把 T4 错写成 E3=T3 已踩过：报 Illegal move 或静默算错题形）
- 读取带超时，区分"慢"和"挂"
"""
import json
import queue
import subprocess
import threading
import itertools

import config

KATAGO = config.KATAGO
MODEL = config.MODEL
CONFIG = config.CONFIG

GC = "ABCDEFGHJKLMNOPQRST"  # 无 I


def gtp(x, y):
    """0-based (x, y) -> GTP 坐标。y+1 因为 GTP 行号 1-based。"""
    return GC[x] + str(y + 1)


def from_gtp(s):
    s = s.strip().upper()
    return GC.index(s[0]), int(s[1:]) - 1


class Engine:
    """单例常驻 KataGo analysis 进程，线程安全（一次一个查询）。"""

    def __init__(self, max_visits=4000):
        self.proc = subprocess.Popen(
            [KATAGO, "analysis", "-model", MODEL, "-config", CONFIG],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self.max_visits = max_visits
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        self._q = queue.Queue()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self):
        for line in self.proc.stdout:
            line = line.strip()
            if line:
                self._q.put(line)

    def query(self, board_size, stones_b, stones_w, moves=None, to_play="B",
              max_visits=None, include_ownership=True):
        """stones_b/stones_w: [(x,y), ...] 0-based。
        moves: [("B","T4"), ...] GTP 字符串（一律用 gtp() 生成）。
        to_play: initialPlayer，指 initialStones 摆完后谁下（默认黑先）。
        返回 {"turns": {turnNumber: response}, ...}，turns[0]=初始局面，
        turns[len(moves)]=forced moves 走完后的局面。"""
        moves = moves or []
        n_turns = len(moves) + 1
        with self._lock:
            qid = f"q{next(self._counter)}"
            q = {
                "id": qid,
                "boardXSize": board_size[0],
                "boardYSize": board_size[1],
                "rules": "Chinese",
                "komi": 7.5,
                "maxVisits": max_visits or self.max_visits,
                "includeOwnership": include_ownership,
                "analyzeTurns": list(range(n_turns)),
                "initialPlayer": to_play,
                "initialStones": [["B", gtp(*s)] for s in stones_b]
                                 + [["W", gtp(*s)] for s in stones_w],
                "moves": [[p, m] for p, m in moves],  # 必带，哪怕空
            }
            self.proc.stdin.write(json.dumps(q) + "\n")
            self.proc.stdin.flush()
            turns = {}
            while True:
                try:
                    line = self._q.get(timeout=180)
                except queue.Empty:
                    raise RuntimeError("KataGo 响应超时（180s），疑似挂起")
                o = json.loads(line)
                if o.get("id") != qid:
                    continue
                if "error" in o:  # error 响应没有 turnNumber，必须显式 raise
                    raise RuntimeError(
                        f"KataGo 错误: {o['error']} — 先检查坐标映射/棋形合法性")
                tn = o.get("turnNumber")
                if tn is not None:
                    turns[tn] = o
                    if tn >= n_turns - 1:
                        break
            return {"turns": turns, "n_moves": len(moves)}

    def close(self):
        self.proc.terminate()


def best_moves(turn_resp, topn=6):
    """从某个 turn 的响应提取候选点 [(move, scoreLead, pv), ...]。"""
    return [(mi["move"], round(mi.get("scoreLead", 0), 1), mi.get("pv", []))
            for mi in turn_resp.get("moveInfos", [])[:topn]]


_engine = None


def get_engine():
    global _engine
    if _engine is None or _engine.proc.poll() is not None:
        _engine = Engine()
    return _engine
