import os
import subprocess
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

    analyzer_proc: subprocess.Popen[str] | None = None
    analyzer_script = root / "scripts" / "p300_analyzer.py"
    if analyzer_script.exists():
        analyzer_proc = subprocess.Popen([sys.executable, str(analyzer_script)])
    else:
        print(f"[run_app] Warning: analyzer script not found: {analyzer_script}")

    from app.main import main as app_main

    try:
        app_main()
    finally:
        if analyzer_proc is not None and analyzer_proc.poll() is None:
            analyzer_proc.terminate()


if __name__ == "__main__":
    main()

