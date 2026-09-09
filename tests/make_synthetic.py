"""合成纸质题图：白底 + 黑网格 + 黑白子 + 蓝笔编号，加旋转/噪声模拟实拍。"""
import cv2
import numpy as np

GC = "ABCDEFGHJKLMNOPQRST"


def make_photo(cols, rows, black, white, digits, out, rot=2.0, noise=True,
               cell=44, margin=40, line_through=False):
    """black/white: [(x,y)] 0-based；digits: [(seq,x,y)]。返回图片路径。
    line_through=True 模拟「线穿白子」印刷：网格线从白子身上穿过
    （真实题册存在这种风格，白子只画描边不盖线）。"""
    w = (cols - 1) * cell + 2 * margin
    h = (rows - 1) * cell + 2 * margin
    img = np.full((h, w, 3), (245, 242, 235), np.uint8)  # 纸色

    def px(x, y):
        return margin + x * cell, margin + y * cell

    for x in range(cols):
        cv2.line(img, px(x, 0), px(x, rows - 1), (30, 30, 30), 2)
    for y in range(rows):
        cv2.line(img, px(0, y), px(cols - 1, y), (30, 30, 30), 2)
    r = int(cell * 0.44)
    for x, y in white:
        cv2.circle(img, px(x, y), r, (250, 250, 250), -1)
        # 描边 2px：1px 描边过 warp 双线性插值会被白芯冲淡消失
        cv2.circle(img, px(x, y), r, (50, 50, 50), 2)
    if line_through:
        # 白子画完后再描一遍线 → 线从白子身上穿过；黑子后画仍盖线。
        # 但白子核心要留亮：真实题册放大看，白子处的线是被白芯盖住/印浅的，
        # 整幅图里白子核心仍是纸色（实测四张真实照片 core-med +2~+27）。
        # 早期版本让线满墨穿过子心，结果 7x7 核心窗（warp 后格距恒 44）里
        # 一半像素是线、core 比空点还暗，白子判据全灭——那是夹具比真实题册
        # 更极端，不是识别的问题（2026-09-09）。
        for x in range(cols):
            cv2.line(img, px(x, 0), px(x, rows - 1), (30, 30, 30), 2)
        for y in range(rows):
            cv2.line(img, px(0, y), px(cols - 1, y), (30, 30, 30), 2)
        for x, y in white:
            cv2.circle(img, px(x, y), max(int(cell * 0.22), 2), (250, 250, 250), -1)
    for x, y in black:
        cv2.circle(img, px(x, y), r, (25, 25, 25), -1)
    for seq, x, y in digits:
        cx, cy = px(x, y)
        cv2.putText(img, str(seq), (cx - 8, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 60, 30), 2)  # 蓝笔 BGR
    if rot:
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=(245, 242, 235))
    if noise:
        img = cv2.GaussianBlur(img, (3, 3), 0.6)
        img = np.clip(img.astype(int) + np.random.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    cv2.imwrite(out, img)
    return out


if __name__ == "__main__":
    import sys
    # 题1 风格：7x5 小棋盘，黑白若干，编号 1/2/3
    make_photo(
        7, 5,
        black=[(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)],
        white=[(1, 2), (2, 2), (3, 2), (4, 2), (3, 3)],
        digits=[(1, 3, 0), (2, 4, 1), (3, 2, 0)],
        out=sys.argv[1] if len(sys.argv) > 1 else "/tmp/synth_p1.jpg",
    )
    print("written")
