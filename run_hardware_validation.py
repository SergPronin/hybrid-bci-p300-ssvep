import os
import runpy
import sys
from pathlib import Path


def _maybe_reexec_in_venv(root: Path) -> None:
    venv_dir = root / ".venv"
    vpy = venv_dir / "bin" / "python"
    if not vpy.exists():
        vpy = venv_dir / "Scripts" / "python.exe"
    if not vpy.exists():
        return

    cur = Path(sys.executable).resolve()
    target = vpy.resolve()
    if cur == target:
        return

    # Re-run under venv Python so imports/dependencies match.
    os.execv(str(target), [str(target), *sys.argv])


def main() -> None:
    root = Path(__file__).resolve().parent
    _maybe_reexec_in_venv(root)
    os.chdir(root)
    sys.path.insert(0, str(root))

    script = root / "scripts" / "hardware_validation.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

