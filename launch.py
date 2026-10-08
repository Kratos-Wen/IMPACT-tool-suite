"""Portable desktop launcher; project assets stay outside source control."""
import argparse, os, runpy, sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Start the IMPACT desktop tool")
    parser.add_argument("--project-profile")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-python")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args, remaining = parser.parse_known_args()
    for key, value in (("IMPACT_PROJECT_PROFILE", args.project_profile),
                       ("IMPACT_SAM2_CHECKPOINT", args.sam_checkpoint),
                       ("IMPACT_SAM2_PYTHON", args.sam_python)):
        if value:
            path = Path(value).expanduser().resolve()
            if not path.is_file():
                parser.error("File does not exist: " + str(path))
            os.environ[key] = str(path)
    if args.device:
        os.environ["IMPACT_SAM2_DEVICE"] = args.device
    os.environ.setdefault("IMPACT_SAM2_DEVICE", "cpu")
    os.environ.setdefault("IMPACT_SAM2_PYTHON", sys.executable)
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))
    sys.argv = [str(root / "app.py"), *remaining]
    runpy.run_path(str(root / "app.py"), run_name="__main__")

if __name__ == "__main__":
    main()
