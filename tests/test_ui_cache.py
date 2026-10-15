import json
from pathlib import Path

from ramanv2.ui.cache import cached_directory_names
from ramanv2.ui.pages.cleaning import _deleted_folders, _deleted_records
from ramanv2.ui.state import invalidate_ui_cache, ui_cache_version


def _write_mapping(path: Path) -> None:
    path.write_text(
        json.dumps({"genera": {"Escherichia": ["EC"]}, "datasets": {"GN": ["Escherichia"]}}),
        encoding="utf-8",
    )


def test_directory_cache_uses_version_for_invalidation(tmp_path: Path) -> None:
    first = tmp_path / "20240101"
    first.mkdir()

    assert cached_directory_names(str(tmp_path), 0) == ("20240101",)

    (tmp_path / "20240102").mkdir()
    assert cached_directory_names(str(tmp_path), 0) == ("20240101",)
    assert cached_directory_names(str(tmp_path), 1) == ("20240101", "20240102")


def test_ui_cache_version_is_process_scoped() -> None:
    """版本号必须与 st.cache_data 同作用域：不能依赖会话状态，否则新会话会重置为 0。"""
    before = ui_cache_version("cleaning")
    invalidate_ui_cache("cleaning")
    assert ui_cache_version("cleaning") == before + 1
    assert ui_cache_version("analysis") == ui_cache_version("analysis")


def test_deleted_records_are_never_served_from_cache(tmp_path: Path) -> None:
    """回归：删除清单必须每次直读磁盘。

    清单曾用 ``@st.cache_data`` + 会话内版本号做键，而 ``st.cache_data`` 跨会话共享：
    新会话（刷新页面、新标签页）版本号回到 0，会复用旧会话留下的清单，对已被移走的文件
    调用绘图从而报 ``WinError 2``。
    """
    data_root = tmp_path / "cosmic_data"
    manual = data_root / "delete" / "manual" / "20240101" / "EC01"
    manual.mkdir(parents=True)
    (manual / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")
    mapping_path = tmp_path / "profile_catalog.json"
    _write_mapping(mapping_path)
    args = (data_root, tmp_path / "missing.xlsx", mapping_path)

    first = _deleted_records(*args)
    assert [record.relative_path for record in first] == ["20240101/EC01/cell1.arc_data"]

    (manual / "cell1.arc_data").unlink()
    assert _deleted_records(*args) == []


def test_deleted_folders_are_never_served_from_cache(tmp_path: Path) -> None:
    """整夹删除清单同样不缓存：恢复整夹后立刻要看到它从列表消失。"""
    archive = tmp_path / "cosmic_data" / "delete" / "folder_delete" / "20240101" / "EC01"
    archive.mkdir(parents=True)
    (archive / "cell1.arc_data").write_text("1 2\n", encoding="utf-8")

    assert [item.relative_path for item in _deleted_folders(tmp_path / "cosmic_data")] == ["20240101/EC01"]

    (archive / "cell1.arc_data").unlink()
    assert _deleted_folders(tmp_path / "cosmic_data") == []
