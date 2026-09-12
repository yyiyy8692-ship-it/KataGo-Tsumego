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
    """返回 (cols, rows, black, white, corner)。"""
    try:
        r = F.recognize_full_board(path)
        if r and r.get("cols", 0) >= 15:
            return (r["cols"], r["rows"], list(r["black"]), list(r["white"]),
                    "FULL")
    except Exception:
        pass                                  # 整盘失败是常态，走局部管线即可
    r = D.recognize(path)
    c = C.locate(path).get("corner") or "BR"
    return r["cols"], r["rows"], list(r["black"]), list(r["white"]), c
