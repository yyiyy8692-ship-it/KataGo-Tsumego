"""KataGo 常驻 analysis 引擎封装。

协议坑（全部踩过，勿改）：
- initialStones 查询也必须带 "moves":[]
- 出错响应没有 turnNumber 字段——读取循环若只等 turnNumber 会永久挂起，必须显式 raise
- ownership 需要 "includeOwnership":true，且在响应顶层
- forced moves 一律用 gtp(x,y) 从坐标元组生成，禁止手拼 GTP 字符串
  （手拼把 T4 错写成 E3=T3 已踩过：报 Illegal move 或静默算错题形）
- 读取带超时，区分"慢"和"挂"
- **ownership 数组是"上下颠倒"的，只翻 y，不翻 x**（2026-09-08 实测 + 官方
  文档确认）。Analysis_Engine.md：数组为行主序，"starting at the top-left of
  the board (e.g. A19) and going to the bottom right (e.g. T1)"。
  即 grid[row][col] 对应 GTP (GC[col], boardYSize - row)。
  故 ownership_grid() 只做 [::-1, :]；写成 [::-1, ::-1] 会多翻 x，读到镜像列。
  着法序列（moveInfos/pv）不受影响，GTP 坐标自洽。
"""
import json
import queue
import subprocess
import threading
import itertools

import numpy as np

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
              max_visits=None, include_ownership=True, komi=None,
              allow_moves=None, avoid_moves=None):
        """stones_b/stones_w: [(x,y), ...] 0-based。
        moves: [("B","T4"), ...] GTP 字符串（一律用 gtp() 生成）。
        to_play: initialPlayer，指 initialStones 摆完后谁下（默认黑先）。
        komi: 默认 7.5（全局对局）。**死活验证必须传 0**——题目不贴目，
              带 7.5 会把"净活但目差落后"的棋块读成劣势，污染 ownership 判定。
        allow_moves / avoid_moves: 限制引擎搜索的着法（行业解死活题的标准手段）。
              格式 [{"player":"B","moves":[...],"untilDepth":20}]，每方最多一个 dict。
              死活题用来把双方锁进局部矩形，防止 AI 脱先弃局部（详见文件头）。
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
                "komi": 7.5 if komi is None else komi,
                "maxVisits": max_visits or self.max_visits,
                "includeOwnership": include_ownership,
                "analyzeTurns": list(range(n_turns)),
                "initialPlayer": to_play,
                "initialStones": [["B", gtp(*s)] for s in stones_b]
                                 + [["W", gtp(*s)] for s in stones_w],
                "moves": [[p, m] for p, m in moves],  # 必带，哪怕空
            }
            if allow_moves:
                q["allowMoves"] = allow_moves
            if avoid_moves:
                q["avoidMoves"] = avoid_moves
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
                    # 必须等**全部** turn 到齐再返回：KataGo 多线程分析会乱序
                    # 返回（实测 numAnalysisThreads=2 时 turns[1] 可能晚于
                    # turns[3]），原先「收到最后一个就 break」会漏掉前面的，
                    # 调用方按 turns[k] 取值直接 KeyError（2026-09-10 实测：
                    # 变化图跑到第5题崩溃）。
                    if len(turns) >= n_turns:
                        break
            return {"turns": turns, "n_moves": len(moves)}

    def close(self):
        self.proc.terminate()


def ownership_grid(resp, board_size):
    """ownership 扁平数组 -> shape (boardYSize, boardXSize)，行序已校正。

    返回 grid[y][x]，y/x 与 gtp(x, y) 使用的 0-based 坐标一一对应。
    协议坑见文件头：KataGo 原始数组是 180° 旋转的，这里 [::-1, ::-1] 翻正。
    """
    w, h = board_size
    return np.array(resp["ownership"], dtype=float).reshape(h, w)[::-1, :]


def move_infos(turn_resp, topn=8):
    """原始 moveInfos（含 visits/winrate/utility/lcb），供选点策略自行比较。

    **排序是 KataGo 自己给的稳健序（LCB，不是裸 scoreLead）**——实测
    （2026-09-11 diag_bestmove.py）：胜率饱和局面里只被访问 1 次的候选会给出
    完全失真的 scoreLead（全局题2 白方 K10 lead=+0.27/vis=1，而实算 -9 目），
    所以按 scoreLead 重排时必须带访问量门槛，否则会挑到噪声手。
    """
    return list(turn_resp.get("moveInfos", [])[:topn])


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
