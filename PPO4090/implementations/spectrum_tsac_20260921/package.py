"""生成独立远程运行包；不携带私钥、虚拟环境、node_modules或历史模型。"""
import argparse
import json
from pathlib import Path
import tarfile
from project_paths import PROJECT_ROOT,SPECTRUM_REPORT_DIR,SPECTRUM_DATA_DIR,SPECTRUM_FRONTEND_DIR
from .audit import sha256
from .artifacts import atomic_json


def build_package(output_name=None):
    SPECTRUM_REPORT_DIR.mkdir(parents=True,exist_ok=True)
    name=output_name or 'SpectrumT-SAC-20260921远程运行包.tar.gz'
    if Path(name).name!=name or not name.endswith('.tar.gz'):raise ValueError('运行包必须为单个tar.gz文件名')
    target=SPECTRUM_REPORT_DIR/name
    files=[PROJECT_ROOT/"project_paths.py",PROJECT_ROOT/"implementations"/"__init__.py",PROJECT_ROOT/"项目说明.md",PROJECT_ROOT/"pytest.ini"]
    if (PROJECT_ROOT/"tests"/"__init__.py").is_file(): files.append(PROJECT_ROOT/"tests"/"__init__.py")
    # 旧源码仅用于兼容审计；新包部署在独立路径，不覆盖远端原项目。
    for folder in (PROJECT_ROOT/"implementations"/"spectrum_tsac_20260921",PROJECT_ROOT/"implementations"/"horizon_tsac_20260920",PROJECT_ROOT/"implementations"/"sac_fyh_io",
                   PROJECT_ROOT/"tests"/"spectrum_tsac_20260921",SPECTRUM_DATA_DIR,PROJECT_ROOT/"data"/"cover_outputs"):
        files.extend(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc",".tmp"))
    for name in ("src","dist","public"):
        path=SPECTRUM_FRONTEND_DIR/name
        if path.is_dir():files.extend(p for p in path.rglob("*") if p.is_file())
    for name in ("package.json","package-lock.json","tsconfig.json","index.html"):
        if (SPECTRUM_FRONTEND_DIR/name).is_file():files.append(SPECTRUM_FRONTEND_DIR/name)
    inventory=[]
    with tarfile.open(target,"w:gz") as archive:
        for path in sorted(set(files)):
            relative=Path("PPO4090")/path.relative_to(PROJECT_ROOT)
            archive.add(path,arcname=relative.as_posix(),recursive=False)
            inventory.append({"path":relative.as_posix(),"sha256":sha256(path),"bytes":path.stat().st_size})
    report={"archive":target.name,"sha256":sha256(target),"files":inventory,"file_count":len(inventory),
            "includes_private_keys":False,"includes_historical_outputs":False,
            "remote_execution_status":"packaged_for_gpu4_validation",
            "note":"运行包不含旧历史模型；远端不要使用本地原产物全量冻结验收作为安装检查"}
    inventory_name=target.name.removesuffix('.tar.gz')+'清单.json' if output_name else 'Spectrum远程运行包清单.json'
    atomic_json(SPECTRUM_REPORT_DIR/inventory_name,report)
    return target,report


def main():
    parser=argparse.ArgumentParser(description="打包新实现、测试、前端和兼容CSV")
    parser.add_argument('--output-name');args=parser.parse_args()
    target,report=build_package(args.output_name)
    print(json.dumps({"archive":str(target),"sha256":report["sha256"],"files":report["file_count"]},ensure_ascii=False))


if __name__=="__main__":main()
