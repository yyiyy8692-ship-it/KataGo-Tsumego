"""网格线位调试图：把 prep_grid 检出的竖线/横线画回原图。"""
import sys

import cv2

import recognition.detect as D

for p in sys.argv[1:]:
    g = D.prep_grid(p)
    img = g["bgr"].copy()
    for x in g["xs"]:
        cv2.line(img, (int(x), 0), (int(x), img.shape[0] - 1), (0, 0, 255), 1)
    for y in g["ys"]:
        cv2.line(img, (0, int(y)), (img.shape[1] - 1, int(y)), (255, 0, 0), 1)
    out = p.replace(".png", "_lines.png")
    cv2.imwrite(out, img)
    print(out, len(g["xs"]), "cols", len(g["ys"]), "rows",
          "cell~%.1f" % (abs(g["xs"][1] - g["xs"][0]) if len(g["xs"]) > 1 else -1))
