from pathlib import Path


project_root = Path(SPECPATH)

analysis = Analysis(
    [str(project_root / "JustFan.py")],
    pathex=[str(project_root)],
    datas=[(str(project_root / "JustFan.ico"), "."), (str(project_root / "JustFan.svg"), ".")],
    excludes=["PyQt5", "PyQt6", "PySide2"],
)
archive = PYZ(analysis.pure)
executable = EXE(
    archive,
    analysis.scripts,
    analysis.binaries,
    analysis.datas,
    [],
    name="JustFan",
    icon=str(project_root / "JustFan.ico"),
    console=False,
    upx=False,
)