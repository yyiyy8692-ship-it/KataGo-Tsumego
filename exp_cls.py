"""实验 v4：detect_full 网格 + classify_by_sweep 分类 → 棋子画回原图。

分类采样点必须精确压在交叉点上，网格错位会立刻在画回图上暴露。
"""
import sys

import cv2
import numpy as np

from recognition import detect as D
from exp_full import detect_full


def main():
    names = sys.argv[1:] or ["page1_b0", "page1_b1", "page1_b2"]
    for name in names:
        p = "photos/%s.png" % name
        bgr = cv2.imread(p)
        bgr, _ = D._deskew_image(bgr)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gg = gray.copy()
        gg[D._blue_mask_strong(bgr) > 0] = 255
        gcls = gray.copy()
        gcls[D._blue_mask(bgr) > 0] = 255
        res = detect_full(gg)
        if res is None:
            print(name, "detect_full 失败")
            continue
        xs, ys = res
        black, white, marks = D.classify_by_sweep(gcls, xs, ys)
        print("%s  黑%2d 白%2d" % (name, len(black), len(white)))
        img = bgr.copy()
        s = float(np.median(np.diff(xs)))
        rad = max(4, int(0.42 * s))
        for (i, j) in black:
            cv2.circle(img, (int(round(xs[i])), int(round(ys[j]))), rad, (0, 0, 255), 2)
        for (i, j) in white:
            cv2.circle(img, (int(round(xs[i])), int(round(ys[j]))), rad, (255, 0, 0), 2)
        cv2.imwrite("dbg_%s_cls.png" % name, img)


if __name__ == "__main__":
    main()
