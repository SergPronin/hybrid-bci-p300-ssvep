import argparse
import fnmatch
import os
import tarfile
from pathlib import Path


DEFAULT_EXCLUDES = [
    ".git",
    ".idea",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "downloads",
    "eggs",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.dylib",
    "*.log",
    "saved_data",
    ".DS_Store",
]


def _should_exclude(rel: str) -> bool:
    rel = rel.replace(os.sep, "/")
    for pat in DEFAULT_EXCLUDES:
        if pat.startswith("*"):
            if fnmatch.fnmatch(rel, pat):
                return True
        else:
            if rel == pat or rel.startswith(pat + "/"):
                return True
    return False


def _add_tar(tar: tarfile.TarFile, root: Path, rel_path: Path) -> None:
    abs_path = root / rel_path
    arcname = root.name + "/" + str(rel_path.as_posix())
    tar.add(str(abs_path), arcname=arcname, recursive=False)


def main() -> int:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Path to archive (tar.gz).")
    args = parser.parse_args()

    out = args.out
    if out is None:
        out = str(root.parent / f"{root.name}_source.tar.gz")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(out_path, "w:gz") as tar:
        for dirpath, dirnames, filenames in os.walk(root):
            dir_abs = Path(dirpath)
            rel_dir = dir_abs.relative_to(root)

            dirnames[:] = [
                d for d in dirnames if not _should_exclude((rel_dir / d).as_posix())
            ]
            for name in filenames:
                rel_file = (rel_dir / name)
                if _should_exclude(rel_file.as_posix()):
                    continue
                _add_tar(tar, root, rel_file)

    print(f"OK. Created: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

