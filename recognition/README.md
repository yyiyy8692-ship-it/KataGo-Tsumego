# recognition — 围棋题图识别模块

把一张印刷死活题的照片变成结构化棋形（黑/白子坐标 + 墙角）+ 一张电子棋盘图。

## 对外 API（只用这一层，内部实现可随时换）

```python
from recognition import recognize_problem, recognize_page

r = recognize_problem("photos/p1.jpg")
r["black"], r["white"]   # [(x, y)] 网格坐标，照片原方向（不翻转）
r["cols"], r["rows"]     # 网格规模（印刷题图常见 7x9）
r["corner"]              # "BR"：墙角，即哪两条边是真棋盘边
r["confidence"]          # 角判定置信度
r["need_confirm"]        # True 时必须人工确认（阈值 corner.CONFIRM_THRESHOLD=0.25）
r["board_svg"]           # 简约电子棋盘 SVG 字符串，可直接嵌网页

page = recognize_page("photos/page.jpg")   # 一图多题：先切分再逐题识别
```

单题想人工指定墙角：`recognize_problem(path, force_corner="BR")`。

## 管线（改之前先读对应函数的文档注释，里面记着踩过的坑）

| 步骤 | 位置 | 做什么 |
|---|---|---|
| 1 | `detect.prep_grid` | 去斜 → 光照归一化 → 检网格（圆心证据 + 细游程投票双路）→ 延伸 → 吸附 |
| 2 | `detect.recognize` | 四角精化 → 逆透视拉正 → 全交叉点扫描分类（黑/白/空） |
| 3 | `corner.locate` | 判墙角：线宽比（主）+ 出头（平局裁决）+ 同册先验 |
| 4 | `detect.split_boards` | 一图多题切分：网格线峰值 → 投票估格距 → 间距断块 |
| 5 | `board.board_svg` | 渲染成简约电子棋盘（深墨蓝石墙，L 形尖角） |

`render.board_svg` 是另一套木板色风格（讲题卡历史样式），两套并存。

## 一图多题：能切，但要看清边界

- **切分本身可靠**：合成 2 题横排 / 3 题横排 / 2x2 版面都切对（块数、阅读顺序都对），
  单题照片不误切（返回原图）。实现见 `split_boards`，旧版 Hough 线段聚类的
  失败记录保留在 `_split_boards_hough` 供对照。
- **识别对像素尺度敏感**：整页拍会让每题变小，格距不足时白子判据（描边环 + 亮核心）
  变脆。建议每题格距 ≥ ~100px（1600 宽的画面里并排 2–3 题比较稳）。
- **写盘必须无损**：白子判据对 JPEG 极敏感（同图另存 jpg 漏 7/7，q100 漏 5/7，
  存 png 不漏），所以 `split_boards` 的裁剪图一律写 **png**。

## 已知边界

- 只支持**印刷题图**（有子必盖线 / 线穿白子两种印刷风格）。真实木盘照片不适用：
  木纹与透视会破坏"线可见即空点"的前提。
- 手写编号识别 2026-09-08 起退休（默认关闭，代码保留 `detect_blue_digits` 供回滚）。
  原因：铅笔读不出、圆珠笔读数仍需人复核，省不掉人工却多一个误判来源。
- q4 残留 1 颗幻影白子 + 1 颗漏判：铅笔手写与白子的「环/核心」特征重叠，
  机器分不开，交给上层画回确认 UI。放宽判据的几次尝试都被真实照片否决，
  记录见 `classify_by_sweep` 注释。
- 角判定的「出头」证据在本册不具区分度（p1 甚至指向反方向），权重压到 0.15
  只做平局裁决；换题册若无区分度应直接摘掉。

## 回归测试

```bash
python tests/run_tests.py          # 识别层全场景（合成图 + 一图多题）
python tests/run_tests.py --api    # 加跑服务 API 全流程（需先启动 5050 端口）
```

真实书页照片的逐子比对脚本在 WorkBuddy 工作区（photos/ + p12_stones.json /
q34_stones.json），不在本仓库——照片不便入库，标准答案由人工核定。
