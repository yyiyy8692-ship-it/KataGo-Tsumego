"""全局配置：路径与端口。全部可用环境变量覆盖，默认值是本机（Mac mini）实测路径。

别人 clone 后只需 export 这几个变量即可跑，不必改代码：
    export KATAGO_BIN=/path/to/katago
    export KATAGO_MODEL=/path/to/model.bin.gz
    export KATAGO_CONFIG=/path/to/analysis_example.cfg
    export TSUME_DATA_DIR=/path/to/go-review/data   # 题库 problems.json 与照片归档目录
    export TSUME_PORT=5050
"""
import os

KATAGO = os.environ.get(
    "KATAGO_BIN", "/Users/yangyang/homebrew/bin/katago")
MODEL = os.environ.get(
    "KATAGO_MODEL",
    "/Users/yangyang/katago/kata1-b18c384nbt-s9996604416-d4316597426.bin.gz")
CONFIG = os.environ.get(
    "KATAGO_CONFIG", "/Users/yangyang/katago/configs/analysis_example.cfg")

_DATA = os.environ.get("TSUME_DATA_DIR", "/Users/yangyang/go-review/data")
PROBLEMS = os.path.join(_DATA, "problems.json")
PHOTOS = os.path.join(_DATA, "photos")

PORT = int(os.environ.get("TSUME_PORT", "5050"))
