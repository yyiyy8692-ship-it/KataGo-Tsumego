"""棋盘 SVG 渲染：画回确认图与正解图共用。
样式与讲题卡一致：木板色底 #E9C992、线 #5F4A2A、黑子 #26221C、白子 #F7F3EA、
手数写子内（黑子白字/白子黑字）、行列坐标标注。
"""
GC = "ABCDEFGHJKLMNOPQRST"


def board_svg(cols, rows, black, white, moves=None, cell=44, margin=34,
              caption="", mark=None):
    """black/white: [(x,y)] 0-based；moves: [(seq,"B"/"W",x,y)] 手数标注；
    mark: (x,y) 画三角要点标记。返回 SVG 字符串。"""
    moves = moves or {}
    move_map = {(x, y): (seq, c) for seq, c, x, y in moves}
    w = (cols - 1) * cell + 2 * margin
    h = (rows - 1) * cell + 2 * margin + (26 if caption else 0)
    p = [f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
         f'style="max-width:100%;background:#E9C992;border-radius:8px">']

    def px(x, y):
        return margin + x * cell, margin + y * cell

    # 网格线
    for x in range(cols):
        a, b = px(x, 0), px(x, rows - 1)
        p.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#5F4A2A" stroke-width="1.2"/>')
    for y in range(rows):
        a, b = px(0, y), px(cols - 1, y)
        p.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="#5F4A2A" stroke-width="1.2"/>')
    # 坐标标注
    for x in range(cols):
        p.append(f'<text x="{px(x,0)[0]}" y="{margin-10}" font-size="13" fill="#5F4A2A" text-anchor="middle">{GC[x]}</text>')
    for y in range(rows):
        p.append(f'<text x="{margin-16}" y="{px(0,y)[1]+4}" font-size="13" fill="#5F4A2A" text-anchor="middle">{rows-y}</text>')

    r = cell * 0.44

    def stone(x, y, color, label=""):
        cx, cy = px(x, y)
        fill = "#26221C" if color == "B" else "#F7F3EA"
        p.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="#5F4A2A" stroke-width="1"/>')
        if label:
            tc = "#F7F3EA" if color == "B" else "#26221C"
            p.append(f'<text x="{cx}" y="{cy+5}" font-size="15" font-weight="bold" fill="{tc}" text-anchor="middle">{label}</text>')

    for x, y in black:
        lbl = str(move_map[(x, y)][0]) if (x, y) in move_map and move_map[(x, y)][1] == "B" else ""
        stone(x, y, "B", lbl)
    for x, y in white:
        lbl = str(move_map[(x, y)][0]) if (x, y) in move_map and move_map[(x, y)][1] == "W" else ""
        stone(x, y, "W", lbl)
    if mark:
        cx, cy = px(*mark)
        p.append(f'<path d="M {cx} {cy-r*0.7} L {cx-r*0.6} {cy+r*0.4} L {cx+r*0.6} {cy+r*0.4} Z" fill="none" stroke="#C0392B" stroke-width="2.5"/>')
    if caption:
        p.append(f'<text x="{w/2}" y="{h-8}" font-size="14" fill="#5F4A2A" text-anchor="middle">{caption}</text>')
    p.append("</svg>")
    return "".join(p)
