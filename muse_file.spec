# -*- mode: python ; coding: utf-8 -*-
import sys
from muse import VERSION

block_cipher = None
added_files = [
         ( 'src/muse/data', 'muse/data' )
         ]

a = Analysis(
    ['src/muse/__main__.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[],
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
    name=f'MUSE_{VERSION}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)


if sys.platform == "darwin":
    app = BUNDLE(exe,
             name='MUSE.app',
             icon=None,
             bundle_identifier=None,
             version=VERSION,
             info_plist={'NSPrincipalClass': 'NSApplication'}
             )