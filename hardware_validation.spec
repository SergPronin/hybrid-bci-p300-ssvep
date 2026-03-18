# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for scripts/hardware_validation.py
#
# Build (Windows):
#   py -3.10 -m PyInstaller --clean --noconfirm hardware_validation.spec
#
# Output:
#   dist/hardware_validation/hardware_validation.exe
#

from PyInstaller.utils.hooks import collect_submodules, collect_data_files, collect_dynamic_libs


block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("pyqtgraph")

datas = []
datas += collect_data_files("pyqtgraph", include_py_files=False)

# pylsl ships a platform-specific liblsl binary; collect it.
binaries = []
binaries += collect_dynamic_libs("pylsl")


a = Analysis(
    ["scripts/hardware_validation.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="hardware_validation",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # keep console to see logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

