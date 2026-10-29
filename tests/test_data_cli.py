from pathlib import Path


from ramanv2.common.arc_data import write_arc_data






def test_build_test_reads_profile_test_root(tmp_path: Path) -> None:
    """独立测试集固定读取 profile 专属的 dataset/<profile>/init_test。"""
    from ramanv2.core.config import InputConfig
    from ramanv2.data import profiles
    from ramanv2.data.builders.stage import build_test

    project = tmp_path / "project"
    wavenumbers = list(range(600, 1801, 10))
    write_arc_data(
        project / "dataset" / "GN" / "init_test" / "ECL20240202_01" / "a.arc_data",
        wavenumbers,
        [float(index) for index in range(len(wavenumbers))],
    )
    # 全局母本中的其他文件夹不应进入 profile 测试集。
    write_arc_data(
        project / "dataset" / "init_test" / "ECL20240203_01" / "b.arc_data",
        wavenumbers,
        [float(index) for index in range(len(wavenumbers))],
    )

    dataset_dir = project / "dataset" / "GN"
    build_test(profiles.get_profile("GN"), dataset_dir, input_config=InputConfig())

    assert (dataset_dir / "test" / "ECL20240202_01" / "a.arc_data").is_file()
    assert not (dataset_dir / "test" / "ECL20240203_01").exists()




