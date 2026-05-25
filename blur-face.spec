# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for blur-face.

Build:
  pyinstaller blur-face.spec
  (one-folder output in dist/blur-face/)
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, collect_data_files

_project_root = Path(SPECPATH)  # directory containing this .spec file

# ── Collect hidden imports ──
_hiddenimports = []

# Recursively collect all ultralytics submodules (handles dynamic imports)
try:
    _hiddenimports += collect_submodules('ultralytics')
except Exception:
    _hiddenimports.append('ultralytics')

# torch internals
_hiddenimports += [
    "torch",
    "torch._C",
    "torchvision",
    "torchvision._C",
    "torchvision.ops",
]

# opencv + numpy
_hiddenimports += [
    "cv2",
    "numpy",
]

# misc
_hiddenimports += [
    "imageio_ffmpeg",
    "imageio_ffmpeg._utils",
    "PIL",
    "tqdm",
    "psutil",
    "pyyaml",
    "requests",
]

# ── Collect data files ──
_datas = []

# Include ultralytics YAML configs etc.
try:
    _datas += collect_data_files('ultralytics')
except Exception:
    pass

# Include ffmpeg binary from imageio-ffmpeg
try:
    import imageio_ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    _datas.append((ffmpeg_exe, "imageio_ffmpeg/bin"))
except Exception:
    pass  # will use system ffmpeg

# ── Exclude heavy modules we don't need ──
_excludes = [
    "matplotlib",
    "pandas",
    "scipy",
    "jupyter",
    "IPython",
    "notebook",
    "tkinter",
    "unittest",
    "test",
    "tests",
]

a = Analysis(
    [str(_project_root / "blur-face.py")],
    pathex=[str(_project_root)],
    binaries=[],
    datas=_datas,
    hiddenimports=_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=_excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="blur-face",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="blur-face",
)
