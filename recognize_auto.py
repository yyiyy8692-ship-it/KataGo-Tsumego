"""统一的单题识别入口：**整盘优先，失败回退局部管线**。

原来这段逻辑写在 solve_page.py 里（CLI 专用），Web 端要用就只能复制一份。
复制的代价是：整盘识别的参数一改，两处的识别行为会悄悄分叉——
这次就是两边用了不同的 recognize（一个返回 5 元组、一个返回 dict），
Web 端全线报 "too many values to unpack"。

所以抽出来，CLI 和 Web 都走这一个函数：
    from recognize_auto import recognize_auto
    cols, rows, black, white, corner = recognize_auto(path)

corner 语义：
    "FULL"  19 路整盘（四边都是边界，无"两墙两开放"的墙角）
    "TL"/"TR"/"BL"/"BR"  局部题的对角墙角（给渲染画 L 形 border 用）
"""
from recognition import detect as D
from recognition import corner as C
import fullboard as F


def recognize_auto(path):
    """返回 (cols, rows, black, white, corner)。

    **必须问 probe_full，不能只看列数**（2026-09-12 修）：
    整盘管线在「一页里的小题图」上也会硬锁出一个 19×19 点阵并返回 cols=19，
    旧写法因此直接把假的整盘结果当成正确答案 —— 微信截图实测报出
    「19×19 黑 87 白 238」（那个截图真正是 7×9）。probe_full 是尺度无关的
    路由判据（能否锁 19×19 + 跨度覆盖≥0.6），三张已人工确认的全局题它全给
    True，误判的截图给 False，正好用来当闸门。

    整盘不是"失败兜底"而是"被证认才用"；局部也识别不出时，最后仍试一次整盘，
    保证不比旧行为更差。
    """
    full_ok = False
    try:
        # probe_full 返回 (is_full, spacing) —— 必须取 [0]，
        # 直接 bool() 包住元组会恒为 True（非空元组真值），闸门形同虚设
        full_ok = bool(F.probe_full(path)[0])
    except Exception:
        full_ok = False
    if full_ok:
        try:
            r = F.recognize_full_board(path)
            if r and r.get("cols", 0) >= 15:
                return (r["cols"], r["rows"], list(r["black"]),
                        list(r["white"]), "FULL")
        except Exception:
            pass                                  # 路由说有，但锁网格失败
    try:
        r = D.recognize(path)
        c = C.locate(path).get("corner") or "BR"
        return r["cols"], r["rows"], list(r["black"]), list(r["white"]), c
    except Exception:
        if full_ok:
            raise                             # 两边都读不出来，报原来的错
        r = F.recognize_full_board(path)      # 局部失败，整盘再兜一次
        return (r["cols"], r["rows"], list(r["black"]), list(r["white"]),
                "FULL")
