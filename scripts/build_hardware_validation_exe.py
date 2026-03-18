import os
import shutil
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    spec_path = os.path.join(repo_root, "hardware_validation.spec")

    if not os.path.exists(spec_path):
        print(f"Spec not found: {spec_path}", file=sys.stderr)
        return 2

    dist_dir = os.path.join(repo_root, "dist")
    build_dir = os.path.join(repo_root, "build")

    for p in (dist_dir, build_dir):
        if os.path.isdir(p):
            shutil.rmtree(p)

    _run([sys.executable, "-m", "PyInstaller", "--clean", "--noconfirm", spec_path])
    print("\nOK. See dist/hardware_validation/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

