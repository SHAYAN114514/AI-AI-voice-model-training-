# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['voice_ai_frontend.py'],
    pathex=[],
    binaries=[],
    datas=[('ai_voice_training_improved.py', '.')],
    hiddenimports=['librosa', 'sklearn', 'matplotlib', 'seaborn', 'joblib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AI语音训练模型',
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
