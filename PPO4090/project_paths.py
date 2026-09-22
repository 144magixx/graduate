"""PPO4090 项目的统一数据与输出路径。

所有路径都锚定到本文件所在的项目根目录，因此训练命令不再依赖当前工作目录。
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
CITIES_DATA_DIR = DATA_DIR / "cities"
COVER_OUTPUT_DIR = DATA_DIR / "cover_outputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
HORIZON_IMPLEMENTATION = "horizon_tsac_20260920"
HORIZON_OUTPUT_DIR = OUTPUTS_DIR / HORIZON_IMPLEMENTATION
HORIZON_DATA_DIR = DATA_DIR / HORIZON_IMPLEMENTATION
HORIZON_FRONTEND_DIR = PROJECT_ROOT / "frontend" / HORIZON_IMPLEMENTATION
HORIZON_REPORT_DIR = PROJECT_ROOT.parent / "第二阶段T-SAC实施记录"

# Spectrum 的源码和运行产物独立；冻结输入及只读看板复用原资源。
SPECTRUM_IMPLEMENTATION = "spectrum_tsac_20260921"
SPECTRUM_OUTPUT_DIR = OUTPUTS_DIR / SPECTRUM_IMPLEMENTATION
SPECTRUM_DATA_DIR = HORIZON_DATA_DIR
SPECTRUM_FRONTEND_DIR = HORIZON_FRONTEND_DIR
SPECTRUM_REPORT_DIR = HORIZON_REPORT_DIR

# Anchor 纯 Transformer 回归链路独立写入；冻结数据与实施记录只读复用。
ANCHOR_IMPLEMENTATION = "anchor_tsac_20260921"
ANCHOR_OUTPUT_DIR = OUTPUTS_DIR / ANCHOR_IMPLEMENTATION
ANCHOR_DATA_DIR = DATA_DIR / ANCHOR_IMPLEMENTATION
ANCHOR_SOURCE_DATA_DIR = HORIZON_DATA_DIR  # 仅只读导入冻结清单；Anchor派生清单写ANCHOR_DATA_DIR。
ANCHOR_FRONTEND_DIR = PROJECT_ROOT / "frontend" / ANCHOR_IMPLEMENTATION
ANCHOR_REPORT_DIR = HORIZON_REPORT_DIR

CITIES_CHINA_CSV = CITIES_DATA_DIR / "cities_china.csv"
CHINA_UNIFORM_POINTS_CSV = CITIES_DATA_DIR / "china_uniform_points.csv"


def get_output_dir(implementation: str, category: str) -> Path:
    """返回并创建某套实现的产物目录。"""

    path = OUTPUTS_DIR / implementation / category
    path.mkdir(parents=True, exist_ok=True)
    return path
