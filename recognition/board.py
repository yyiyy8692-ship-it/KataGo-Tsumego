"""电子棋盘渲染：把识别结果画成简约风格的 SVG 棋盘图（画回确认 / 讲题卡用）。

风格（2026-09-09 用户定稿，参照其配色样图）：
- 白底、浅灰细网格线；开放边与内部线一视同仁，都是普通网格
- 石墙（真棋盘边）＝深墨蓝粗线，两条边用**单条 L 形路径**画、拐角尖角拼接
  （两条线各自平移出头会在拐角留断口，必须共用拐点 + miter）
- 黑子近黑、白子纯白描浅灰边，无坐标无杂物，一眼可核对
- 图下小字：题名 + 「墙角 右 + 底 ｜ 黑 x · 白 y」

另有一套木板色风格在 render.py（讲题卡历史样式），两套并存由调用方选。
"""
import html

C_BG = "#FFFFFF"          # 棋盘底
C_GRID = "#C9CDD1"        # 网格线（浅灰）
C_WALL = "#2C4E72"        # 石墙（深墨蓝；备选木棕 #8A5E3C、炭灰 #4A4A4A）
C_TEXT = "#2B2B2B"        # 主文字
C_SUB = "#2C4E72"         # 副文字（与墙同色系）
C_BLACK = "#2A2620"       # 黑子
C_WHITE = "#FFFFFF"       # 白子
C_WHITE_STROKE = "#C0C4C8"

CORNER_LABEL = {"TL": "左 + 顶", "TR": "右 + 顶",
                "BL": "左 + 底", "BR": "右 + 底"}
WALL_EDGES = {"TL": ("L", "T"), "TR": ("R", "T"),
              "BL": ("L", "B"), "BR": ("R", "B")}


def board_svg(name, black, white, corner, cols=7, rows=9,
              cell=46, margin=44):
    """black/white: [(x,y)] 0-based，照片原方向（不翻转，方便与书页对照）。

    返回 SVG 字符串。corner ∈ TL/TR/BL/BR，决定哪两条边画成石墙。
    """
    over = 7                      # 墙线端点出头像素
    w = (cols - 1) * cell + 2 * margin
    h = (rows - 1) * cell + 2 * margin + 64   # 底部留两行小字

    def px(x, y):
        return margin + x * cell, margin + y * cell

    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:100%;background:{C_BG}">']

    # 1) 网格线（含四条边——开放边就是普通网格线，由墙线覆盖出真边）
    for x in range(cols):
        ax, ay = px(x, 0)
        bx, by = px(x, rows - 1)
        p.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" '
                 f'stroke="{C_GRID}" stroke-width="2"/>')
    for y in range(rows):
        ax, ay = px(0, y)
        bx, by = px(cols - 1, y)
        p.append(f'<line x1="{ax}" y1="{ay}" x2="{bx}" y2="{by}" '
                 f'stroke="{C_GRID}" stroke-width="2"/>')

    # 2) 石墙：一条 L 形路径，拐角处尖角拼接（miter），两条边严丝合缝
    wl, wt = WALL_EDGES[corner]
    sw = cell * 0.16
    xc = cols - 1 if "R" in (wl, wt) else 0          # 墙角的列
    yc = rows - 1 if "B" in (wl, wt) else 0          # 墙角的行
    xf = 0 if xc == cols - 1 else cols - 1           # 横边的另一端
    yf = 0 if yc == rows - 1 else rows - 1           # 竖边的另一端
    ax, ay = px(xc, yf)
    bx, by = px(xf, yc)
    kx, ky = px(xc, yc)
    ay += -over if yf == 0 else over                 # 竖边向外出头
    bx += -over if xf == 0 else over                 # 横边向外出头
    p.append(f'<path d="M {ax} {ay} L {kx} {ky} L {bx} {by}" fill="none" '
             f'stroke="{C_WALL}" stroke-width="{sw:.1f}" stroke-linecap="round" '
             f'stroke-linejoin="miter"/>')

    # 3) 棋子
    r = cell * 0.44
    for x, y in black:
        cx, cy = px(x, y)
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="{C_BLACK}"/>')
    for x, y in white:
        cx, cy = px(x, y)
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{r:.1f}" fill="{C_WHITE}" '
                 f'stroke="{C_WHITE_STROKE}" stroke-width="1.5"/>')

    # 4) 底部小字（排版同参考样图：主标题深色 + 副行与墙同色）
    ty = h - 40
    p.append(f'<text x="{w / 2}" y="{ty}" font-size="17" font-weight="600" '
             f'fill="{C_TEXT}" text-anchor="middle" '
             f'font-family="PingFang SC, sans-serif">{html.escape(name)}</text>')
    p.append(f'<text x="{w / 2}" y="{ty + 26}" font-size="14" '
             f'fill="{C_SUB}" text-anchor="middle" '
             f'font-family="PingFang SC, sans-serif">'
             f'墙角 {CORNER_LABEL[corner]} ｜ 黑 {len(black)} · 白 {len(white)}</text>')
    p.append("</svg>")
    return "".join(p)
