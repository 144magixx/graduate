"""全项目AST与非训练导入检查；不会导入历史训练入口。"""
import ast
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
from project_paths import PROJECT_ROOT,HORIZON_REPORT_DIR
from .artifacts import atomic_json
from .audit import verify as verify_frozen


def main():
    code_files=[PROJECT_ROOT/"project_paths.py"]+list((PROJECT_ROOT/"implementations").rglob("*.py"))+list((PROJECT_ROOT/"tests").rglob("*.py"))
    failures=[]
    for path in code_files:
        try: ast.parse(path.read_text(encoding="utf-8-sig"),filename=str(path))
        except Exception as error: failures.append({"path":str(path),"error":str(error)})
    if failures: raise RuntimeError(failures)
    old=[]
    new=[]
    for path in (PROJECT_ROOT/"implementations").rglob("*.py"):
        name=".".join(path.relative_to(PROJECT_ROOT).with_suffix("").parts)
        if path.name=="__init__.py":name=name.rsplit(".",1)[0]
        if "horizon_tsac_20260920" in path.parts:
            new.append(name)
        elif path.name.startswith("Environment") or path.name in ("sac_fyh_IO.py","sac_transformer_tok.py","sac_transformer_p.py"):
            old.append(name)
    imports=[]
    for category,modules in (("legacy_non_training",old),("new_including_guarded_entries",new)):
        script="import importlib,json; modules="+repr(sorted(set(modules)))+"; [importlib.import_module(x) for x in modules]; print(json.dumps({'imported':modules}))"
        started=time.perf_counter()
        result=subprocess.run([sys.executable,"-B","-c",script],cwd=PROJECT_ROOT,capture_output=True,text=True,encoding="utf-8",timeout=60)
        if result.returncode: raise RuntimeError(result.stderr)
        imports.append({"category":category,"modules":sorted(set(modules)),"seconds":time.perf_counter()-started,"stderr":result.stderr})
    result={"ast_passed":len(code_files),"import_groups":imports,"baseline":verify_frozen(),
            "old_training_entries_imported":False,"new_entrypoints_guarded":True}
    HORIZON_REPORT_DIR.mkdir(parents=True,exist_ok=True)
    atomic_json(HORIZON_REPORT_DIR/"静态检查与导入验收.json",result)
    print(json.dumps({"ast_passed":len(code_files),"imports_passed":sum(len(x["modules"]) for x in imports),"baseline":result["baseline"]},ensure_ascii=False))


if __name__=="__main__":main()
