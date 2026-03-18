import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)

def _venv_python(venv_dir: Path) -> Path:
    # macOS/Linux: <venv>/bin/python
    # Windows: <venv>/Scripts/python.exe
    posix = venv_dir / "bin" / "python"
    win = venv_dir / "Scripts" / "python.exe"
    return win if win.exists() else posix


def _which_or_none(exe: str) -> str | None:
    resolved = shutil.which(exe)
    return resolved


def main() -> int:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-dir", default=str(root / ".venv"))
    parser.add_argument("--requirements", default=str(root / "requirements.txt"))
    parser.add_argument("--python", default="python3.10", help="Python interpreter to use for venv (default: python3.10).")
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()

    venv_dir = Path(args.venv_dir)
    requirements = Path(args.requirements)
    python_exe = _which_or_none(args.python) or _which_or_none("python") or sys.executable

    if not requirements.exists():
        raise FileNotFoundError(f"requirements not found: {requirements}")

    # Create venv using the requested interpreter (so dependency pins work).
    if not _venv_python(venv_dir).exists():
        if not _which_or_none(args.python):
            print(
                f"WARNING: requested interpreter '{args.python}' not found; falling back to '{python_exe}'.",
                file=sys.stderr,
            )
        _run([python_exe, "-m", "venv", str(venv_dir)])

    py = _venv_python(venv_dir)
    if not args.no_install:
        _run([str(py), "-m", "pip", "install", "-r", str(requirements)])

    print("OK. If you want to run commands manually, activate the venv:")
    print(f"  {venv_dir}/bin/activate  # macOS/Linux")
    print(f"  {venv_dir}\\Scripts\\activate  # Windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

