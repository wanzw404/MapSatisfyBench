"""统一路径配置模块。

支持通过环境变量 EVAL_SERVICE_DATA_DIR 指定 data 目录位置，
否则通过 __file__ 自动推导项目根目录。

远程服务器部署时，如果目录结构不同，可设置环境变量：
    export EVAL_SERVICE_DATA_DIR=/home/admin/amap-eval-service/target/amap-eval-service/data
"""

import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def _find_data_dir() -> Path:
    """查找 data 目录位置。

    查找优先级：
    1. 环境变量 EVAL_SERVICE_DATA_DIR
    2. 通过 app/paths.py 推导项目根目录（向上2层）
    3. 再向上多找一层（兼容某些嵌套部署结构）
    """
    # 优先级1：环境变量
    env_dir = os.environ.get("EVAL_SERVICE_DATA_DIR")
    if env_dir:
        data_dir = Path(env_dir)
        logger.info(f"使用环境变量指定的 data 目录: {data_dir}")
        return data_dir

    # 优先级2：通过当前文件推导（app/paths.py -> app/ -> 项目根）
    _file = Path(__file__).resolve()
    project_root = _file.parent.parent  # paths.py 在 app/ 下
    data_dir = project_root / "data"
    if data_dir.exists():
        logger.debug(f"通过 __file__ 推导 data 目录: {data_dir}")
        return data_dir

    # 优先级3：向上再多找一层（兼容某些部署结构）
    project_root = _file.parent.parent.parent
    data_dir = project_root / "data"
    if data_dir.exists():
        logger.debug(f"通过 __file__ 推导 data 目录（向上多层）: {data_dir}")
        return data_dir

    # 兜底：返回默认推导路径（调用方会按需创建子目录）
    default = _file.parent.parent / "data"
    logger.warning(f"未找到已存在的 data 目录，使用默认路径: {default}")
    return default


# 统一 data 目录
DATA_DIR = _find_data_dir()

# 各子目录
INPUT_DIR = DATA_DIR / "inputs"
OUTPUT_DIR = DATA_DIR / "outputs" / "simulator_res"
SIMULATOR_RES_DIR = DATA_DIR / "outputs" / "simulator_res"
EVALUATION_RES_DIR = DATA_DIR / "outputs" / "evaluation_res"
NEW_DATA_DIR = DATA_DIR / "sandbox" / "new_data"
EXACT_SEARCH_DIR = DATA_DIR / "sandbox" / "mock_data"
