import argparse
import subprocess
import sys
import venv
from pathlib import Path


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _venv_python(venv_dir: Path) -> Path:
    posix = venv_dir / "bin" / "python"
    win = venv_dir / "Scripts" / "python.exe"
    return win if win.exists() else posix


def main() -> int:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--venv-dir", default=str(root / ".venv"))
    parser.add_argument("--requirements", default=str(root / "requirements.txt"))
    parser.add_argument("--no-install", action="store_true")
    args = parser.parse_args()

    venv_dir = Path(args.venv_dir)
    requirements = Path(args.requirements)

    if not requirements.exists():
        raise FileNotFoundError(f"requirements not found: {requirements}")

    if not _venv_python(venv_dir).exists():
        builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False)
        builder.create(str(venv_dir))

    py = _venv_python(venv_dir)
    if not args.no_install:
        _run([str(py), "-m", "pip", "install", "-r", str(requirements)])

    print("OK. If you want to run commands manually, activate the venv:")
    print(f"  {venv_dir}/bin/activate  # macOS/Linux")
    print(f"  {venv_dir}\\Scripts\\activate  # Windows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

