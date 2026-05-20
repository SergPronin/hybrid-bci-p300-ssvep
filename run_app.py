import argparse
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

    parser = argparse.ArgumentParser(description="Запуск стимулятора плиток (PsychoPy) и опций.")
    parser.add_argument(
        "--auto-random-protocol",
        action="store_true",
        help="Без экранных кнопок: сразу trial, новая случайная цель после каждого trial_end.",
    )
    parser.add_argument(
        "--no-analyzer",
        action="store_true",
        help="Не поднимать отдельно scripts/p300_analyzer.py (для protocol_runner и т.п.).",
    )
    parser.add_argument(
        "--inter-trial-s",
        type=float,
        default=1.0,
        help="Пауза между trial в режиме --auto-random-protocol, сек.",
    )
    args = parser.parse_args()

    analyzer_proc: subprocess.Popen[str] | None = None
    analyzer_script = root / "scripts" / "p300_analyzer.py"
    if not args.no_analyzer and analyzer_script.exists():
        analyzer_proc = subprocess.Popen([sys.executable, str(analyzer_script)])
    elif not args.no_analyzer:
        print(f"[run_app] Warning: analyzer script not found: {analyzer_script}")

    from app.main import main as app_main

    app_main(
        auto_random_trials=bool(args.auto_random_protocol),
        inter_trial_s=float(args.inter_trial_s),
    )
    if analyzer_proc is not None and analyzer_proc.poll() is None:
        print(
            "[run_app] Stimulator closed. P300 analyzer keeps running in a separate window/process."
        )


if __name__ == "__main__":
    main()

