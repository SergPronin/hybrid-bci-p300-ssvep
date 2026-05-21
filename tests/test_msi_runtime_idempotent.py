"""Повторный вызов _configure_coreclr не должен падать после первой настройки runtime."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def test_configure_coreclr_skips_when_runtime_already_set() -> None:
    import sys

    from scripts import test_msi_import as tmi

    sys.modules["clr"] = MagicMock()
    fake_runtime = MagicMock()
    try:
        with patch("pythonnet._RUNTIME", fake_runtime), patch("pythonnet._LOADED", False):
            with patch("pythonnet.set_runtime") as mock_set:
                tmi._configure_coreclr(Path("/tmp/dotnet"))
                mock_set.assert_not_called()

        with patch("pythonnet._RUNTIME", None), patch("pythonnet._LOADED", True):
            with patch("pythonnet.set_runtime") as mock_set:
                tmi._configure_coreclr(Path("/tmp/dotnet"))
                mock_set.assert_not_called()
    finally:
        sys.modules.pop("clr", None)


def test_load_msi_runtime_returns_cached_instance() -> None:
    from scripts import test_msi_exec as tme

    cached = (MagicMock(name="msi"), Path("/msi-res"), Path("/dotnet"))
    tme._MSI_RUNTIME_CACHE = cached
    try:
        assert tme.load_msi_runtime() is cached
    finally:
        tme._MSI_RUNTIME_CACHE = None
