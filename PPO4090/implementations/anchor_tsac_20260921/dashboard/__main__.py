"""python -m implementations.anchor_tsac_20260921.dashboard --port 8765"""
import argparse
import tempfile


def main():
    parser = argparse.ArgumentParser(description="Anchor T-SAC 只读看板（不启动训练）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-root", "--run-root", dest="runs_root")
    parser.add_argument("--datasets-root")
    parser.add_argument("--static-root")
    parser.add_argument("--demo", action="store_true", help="仅启动临时工程示例；不加载 CSV、模型或权重")
    args = parser.parse_args()
    import uvicorn
    from .api import create_app
    if args.demo:
        from .demo import create_demo_run
        with tempfile.TemporaryDirectory(prefix="anchor-dashboard-demo-") as root:
            create_demo_run(root)
            uvicorn.run(create_app(root, root, args.static_root), host=args.host, port=args.port)
    else:
        uvicorn.run(create_app(args.runs_root, args.datasets_root, args.static_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
