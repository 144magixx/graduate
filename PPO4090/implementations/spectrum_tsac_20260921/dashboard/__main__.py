"""python -m implementations.spectrum_tsac_20260921.dashboard --port 8765"""
import argparse


def main():
    parser = argparse.ArgumentParser(description="Spectrum T-SAC 只读看板（不启动训练）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--runs-root", "--run-root", dest="runs_root")
    parser.add_argument("--datasets-root")
    parser.add_argument("--static-root")
    args = parser.parse_args()
    import uvicorn
    from .api import create_app
    uvicorn.run(create_app(args.runs_root, args.datasets_root, args.static_root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
