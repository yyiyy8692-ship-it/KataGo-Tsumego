# 死活题拍照批改 App（KataGo 驱动）

给**不会下围棋的家长**用的死活题批改工具：孩子把做好的纸质死活题拍一张照片上传，系统识别棋形和孩子手写答案，用 KataGo 验证正解，输出批改结论 + 四步启发式讲解 + 可打印讲题卡。

> 使用场景：孩子业余 4 段，每天做死活题，但家长不会围棋、无法当场批改，也不想只给"对/错"这种没有营养的反馈。

- 手机浏览器访问（局域网内），无需安装 App
- 后端：Python + Flask + OpenCV + KataGo（常驻 analysis 进程）
- 详细设计思路见 [`docs/DESIGN.md`](docs/DESIGN.md)

---

## 流程（四步）

| 步骤 | 做什么 | 为什么必须 |
|---|---|---|
| ① 拍照 | 拍题图 + **必选题型**（黑先杀/黑先活/白先杀/白先活） | 系统不猜先后手——杀/活是一体两面，猜错方向整个判定反了 |
| ② 核对棋形 | 把识别结果画回棋盘，点交叉点可改子（空→黑→白循环） | **硬性门禁**：识别差一子就能翻转死活结论，未确认不批改 |
| ③ 录答案 | 自动识别手写蓝字编号，按 1→2→3 顺序点选确认/修正 | 手写编号读不准时宁可标 `?` 交给人工，不冤枉孩子 |
| ④ 批改 | 正解 + 孩子变化逐手评分 + 一致性警报 + 四步启发阶梯 | 反馈的是"怎么想"，不是"答案是什么" |

## 目录结构

```
tsume-app/
├── config.py              # 路径/端口配置（全部支持环境变量覆盖）
├── engine/katago.py       # KataGo 常驻 analysis 进程封装（协议坑已固化）
├── recognition/
│   ├── detect.py          # 照片 → 网格 + 黑白子 + 手写编号
│   └── render.py          # 棋盘 SVG 绘制
├── grader/
│   ├── grader.py          # 批改：正解、逐手评分、一致性警报、启发阶梯
│   └── archive.py         # 批改结果归档进题库（自动备份 + 写后校验）
├── web/
│   ├── app.py             # Flask 路由
│   ├── templates/         # index.html（四步向导）、report.html（讲题卡）
│   └── static/            # app.js、style.css
├── tests/                 # 合成题图生成器 + 14 项自动化测试
├── deploy/                # macOS 开机自启 plist
├── start.sh
└── requirements.txt
```

## 快速开始

**1. 依赖**

```bash
pip install -r requirements.txt
```

还需要本地 KataGo（[官方发布页](https://github.com/lightvector/KataGo/releases)）与一个模型文件。

**2. 配置**（`config.py`，全部可用环境变量覆盖，不用改代码）

```bash
export KATAGO_BIN=/path/to/katago
export KATAGO_MODEL=/path/to/model.bin.gz
export KATAGO_CONFIG=/path/to/analysis_example.cfg
export TSUME_DATA_DIR=/path/to/data        # 题库 problems.json、照片归档目录
export TSUME_PORT=5050
```

**3. 启动**

```bash
./start.sh                 # 或 python web/app.py
```

手机连同一 Wi-Fi，浏览器打开 `http://<本机局域网 IP>:5050`。

**4. 开机自启（macOS）**

先把 `deploy/com.tsume.app.plist` 里的 `CHANGE_ME` 换成你的用户名与实际项目路径，然后：

```bash
cp deploy/com.tsume.app.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.tsume.app.plist
```

## 测试

```bash
cd tests
python run_tests.py        # 识别层 7 项
python run_tests.py --api  # + API 层 7 项（需服务已启动）
```

测试用合成题图（含旋转、噪声、透视扰动）每次重新生成，可用于稳定性统计。
批改链路的归档测试用临时题库，不会污染生产 `problems.json`。

当前基线（20 轮×2 组合成图统计）：棋子识别 100/100（含 10% 透视 + 光照梯度 + 5° 旋转的加强组）；手写编号正视角 20/20、透视下读错 0/60（读不准标 `?` 走人工确认，是设计行为）。

识别管线对标 GitHub 围棋拍照识别最佳实践（四角精化 → 逆透视拉正 → 等距常量网格扫描分类），详见 [`docs/DESIGN.md`](docs/DESIGN.md) 第 4 节。

## 已知限制

- 识别面向**印刷题图**（"有子必盖线"判据），真实木盘照片不适用。
- 手写数字识别保守：低置信一律标 `?` 走人工确认（透视下约两成数字会标 `?`）。
- 仅支持蓝笔编号（HSV H 85~135）；换笔色需调阈值。

## 后续（V2）

- 孩子模式：阶梯式对话引导，不直接报答案
- 错题本打通与周期统计
- 真实作业照片实测调参
