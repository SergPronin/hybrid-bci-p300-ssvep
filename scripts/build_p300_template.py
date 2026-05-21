#!/usr/bin/env python3
"""Построить и сохранить P300-эталон из калибровочных CSV файлов."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from p300_analysis.p300_template import build_p300_template, save_p300_template


def _collect_files(paths: List[str]) -> List[Path]:
    """Собрать CSV файлы из списка путей."""
    files: List[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            # Искать CSV файлы в директории
            files.extend(sorted(pp.glob("*_continuous.csv")))
        elif pp.exists():
            files.append(pp)
        else:
            print(f"[WARN] не найден: {pp}", file=sys.stderr)
    return sorted(set(files))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить P300-эталон из калибровочных CSV файлов. "
                    "Эталон используется для классификации плиток методом CCA."
    )
    parser.add_argument(
        "paths", nargs="+",
        help="CSV файлы или директории с калибровочными данными"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="p300_template.npz",
        help="Путь для сохранения эталона (по умолчанию: p300_template.npz)"
    )
    parser.add_argument(
        "--baseline-ms", type=int, default=100,
        help="Длина baseline окна в миллисекундах (по умолчанию: 100)"
    )
    parser.add_argument(
        "--artifact-uv", type=float, default=60.0,
        help="Порог отклонения артефактов в микровольтах (по умолчанию: 60.0)"
    )
    parser.add_argument(
        "--car", action="store_true",
        help="Применить Common Average Reference (CAR) при построении эталона"
    )

    args = parser.parse_args()

    # Собрать файлы
    files = _collect_files(args.paths)
    if not files:
        print("❌ Не найдены CSV файлы.", file=sys.stderr)
        sys.exit(1)

    print(f"📂 Найдено {len(files)} файлов калибровки:")
    for f in files:
        print(f"  - {f.name}")
    print()

    # Построить эталон
    try:
        print("🔧 Построение P300-эталона...")
        template, time_ms, stim_templates = build_p300_template(
            files,
            baseline_ms=int(args.baseline_ms),
            artifact_uv=float(args.artifact_uv),
            use_car=bool(args.car),
        )
        print(f"✓ Эталон построен: форма {template.shape}")
        print(f"✓ Найдено {len(stim_templates)} типов стимулов:")
        for stim_key in sorted(stim_templates.keys()):
            print(f"  - {stim_key}: форма {stim_templates[stim_key].shape}")
        print()

        # Сохранить эталон
        output_path = Path(args.output)
        print(f"💾 Сохранение в {output_path}...")
        save_p300_template(template, time_ms, stim_templates, output_path)
        print(f"✓ Эталон сохранен в:")
        print(f"  - {output_path}")
        print(f"  - {output_path.with_suffix('.json')} (метаинформация)")
        print()
        print("✅ Готово! Используйте эталон в GUI через меню 'Загрузить P300-эталон'")

    except Exception as e:
        print(f"❌ Ошибка: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
