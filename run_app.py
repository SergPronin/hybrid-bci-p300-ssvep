import os
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root))

    from app.main import main as app_main

    app_main()


if __name__ == "__main__":
    main()

