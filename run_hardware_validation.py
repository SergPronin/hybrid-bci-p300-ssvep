import os
import runpy
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))

    script = root / "scripts" / "hardware_validation.py"
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()

