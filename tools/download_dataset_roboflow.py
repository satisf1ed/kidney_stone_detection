import argparse
import os
from pathlib import Path
from roboflow import Roboflow

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data", help="Output directory")
    ap.add_argument("--project", type=str, required=True, help="Roboflow project slug")
    ap.add_argument("--version", type=int, required=True, help="Dataset version")
    ap.add_argument("--format", type=str, default="yolov8", help="Export format, e.g. yolov8")
    ap.add_argument("--workspace", type=str, default=None, help="Optional workspace slug")
    args = ap.parse_args()

    api_key = os.environ.get("ROBOFLOW_API_KEY")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(args.workspace) if args.workspace else rf.workspace()
    proj = ws.project(args.project)
    ver = proj.version(args.version)
    ds = ver.download(args.format, location=str(out_dir))

    print("Downloaded to:", ds.location)
    print("data.yaml:", Path(ds.location) / "data.yaml")

if __name__ == "__main__":
    main()
