"""在本地 Matplotlib 窗口中浏览并人工审查 cosmic_data 光谱。"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg", force=True)

import matplotlib.pyplot as plt
from matplotlib import font_manager

plt.ioff()


def resolve_project_dir() -> Path:
    """根据脚本位置定位包含 ramanv2 与 dataset 的项目根目录。"""
    project_dir = Path(__file__).resolve().parents[1]
    if (project_dir / "ramanv2").is_dir() and (project_dir / "dataset").is_dir():
        return project_dir
    raise FileNotFoundError("未找到同时包含 ramanv2 和 dataset 的项目根目录。")


PROJECT_DIR = resolve_project_dir()
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from ramanv2.data.io import read_arc_data


DEFAULT_DATA_DIR = PROJECT_DIR / "dataset" / "CSdata" / "cosmic_data"
REVIEW_LOG_NAME = "reviewed_dates.log"
DELETED_LOG_NAME = "deleted_spectra.log"
SPECTRA_PER_PAGE = 9


def configure_chinese_font() -> None:
    """设置可用的中文字体，避免图标题与状态文字缺字。"""
    font_paths = (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    )
    for font_path in font_paths:
        if font_path.is_file():
            font_manager.fontManager.addfont(str(font_path))

    font_names = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in ("Microsoft YaHei", "SimHei"):
        if font_name in font_names:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False


def resolve_data_dir(data_dir_value: str) -> Path:
    """解析命令行给出的 cosmic_data 目录。"""
    data_dir = Path(data_dir_value)
    if not data_dir.is_absolute():
        data_dir = PROJECT_DIR / data_dir
    if not data_dir.is_dir():
        raise NotADirectoryError(f"cosmic_data 目录不存在：{data_dir}")
    return data_dir


def parse_natural_sort_key(value: str) -> tuple[tuple[int, int | str], ...]:
    """将文本拆为数字与文本片段，使文件名按实际数字顺序排列。"""
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
    )


def list_spectrum_dirs(date_dir: Path) -> list[Path]:
    """列出日期目录下包含 .arc_data 的直接子文件夹。"""
    return [
        spectrum_dir
        for spectrum_dir in sorted(
            date_dir.iterdir(), key=lambda path: parse_natural_sort_key(path.name)
        )
        if spectrum_dir.is_dir() and any(spectrum_dir.rglob("*.arc_data"))
    ]


def load_reviewed_date_names(log_path: Path) -> set[str]:
    """读取已经完整审查的日期名称。"""
    if not log_path.is_file():
        return set()
    return {
        line.split("\t", 1)[0]
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def write_reviewed_date(log_path: Path, date_name: str) -> None:
    """记录完成审查的日期与记录时间。"""
    recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"{date_name}\t{recorded_at}\n")


def write_deleted_spectrum(log_path: Path, data_dir: Path, spectrum_path: Path) -> None:
    """记录成功删除的光谱时间与相对数据根目录路径。"""
    deleted_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    relative_path = spectrum_path.relative_to(data_dir).as_posix()
    rows = log_path.read_text(encoding="utf-8").splitlines() if log_path.is_file() else []
    rows.append(f"{deleted_at}\t{relative_path}")
    rows.sort(key=lambda row: parse_natural_sort_key(row.split("\t", 1)[1]))
    log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def list_pending_date_dirs(data_dir: Path, reviewed_date_names: set[str]) -> list[Path]:
    """列出仍需审查且包含光谱子文件夹的日期目录。"""
    return [
        date_dir
        for date_dir in sorted(
            data_dir.iterdir(), key=lambda path: parse_natural_sort_key(path.name)
        )
        if date_dir.is_dir()
        and date_dir.name not in reviewed_date_names
        and list_spectrum_dirs(date_dir)
    ]


class CosmicDataBrowser:
    """管理本地窗口中的日期、子文件夹浏览和单谱删除操作。"""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.review_log_path = data_dir / REVIEW_LOG_NAME
        self.deleted_log_path = data_dir / DELETED_LOG_NAME
        self.reviewed_date_names = load_reviewed_date_names(self.review_log_path)
        self.date_dirs = list_pending_date_dirs(data_dir, self.reviewed_date_names)
        if not self.date_dirs:
            raise FileNotFoundError(
                f"没有待审查日期；如需重新浏览，请编辑日志：{self.review_log_path}"
            )

        self.date_folder_dirs = {
            date_dir: list_spectrum_dirs(date_dir) for date_dir in self.date_dirs
        }
        self.date_index = 0
        self.folder_index = 0
        self.page_index = 0
        self.reviewed_spectrum_paths: set[Path] = set()
        self.selected_path: Path | None = None
        self.selected_line = None
        self.figure = plt.figure(figsize=(12, 7), dpi=90)
        self.figure.canvas.mpl_connect("pick_event", self.select_spectrum)
        self.figure.canvas.mpl_connect("key_press_event", self.change_browse_position)
        self.draw_current_folder()

    def show(self) -> None:
        """显示浏览窗口并输出可用操作说明。"""
        print("操作：PageDown 翻到下一页，末页自动进入下一个子文件夹或日期。")
        print("PageUp 切换当前子文件夹的上一页。")
        print("单击曲线可选中；按 Enter 删除对应 .arc_data 原文件。")
        plt.show(block=True)

    def draw_current_folder(self) -> None:
        """将当前子文件夹的一页光谱排列为独立的大子图。"""
        date_dir = self.date_dirs[self.date_index]
        folder_dirs = self.date_folder_dirs[date_dir]
        self.folder_index %= len(folder_dirs)
        folder_dir = folder_dirs[self.folder_index]
        spectrum_paths = sorted(
            folder_dir.rglob("*.arc_data"),
            key=lambda path: parse_natural_sort_key(path.name),
        )
        page_count = max(1, (len(spectrum_paths) + SPECTRA_PER_PAGE - 1) // SPECTRA_PER_PAGE)
        self.page_index %= page_count
        page_start = self.page_index * SPECTRA_PER_PAGE
        page_paths = spectrum_paths[page_start : page_start + SPECTRA_PER_PAGE]

        self.selected_path = None
        self.selected_line = None
        spectra = []
        for spectrum_path in page_paths:
            wavenumbers, intensities = read_arc_data(spectrum_path)
            if wavenumbers.size == 0 or intensities.size == 0:
                continue
            spectra.append((spectrum_path, wavenumbers, intensities))
        self.reviewed_spectrum_paths.update(page_paths)

        self.figure.clear()
        visible_count = len(spectra)
        self.figure.suptitle(
            f"{date_dir.name} / {folder_dir.name} | "
            f"日期 {self.date_index + 1}/{len(self.date_dirs)}，"
            f"子文件夹 {self.folder_index + 1}/{len(folder_dirs)}，"
            f"第 {self.page_index + 1}/{page_count} 页，光谱 {visible_count}/{len(spectrum_paths)} 条"
        )
        if not spectra:
            axis = self.figure.add_subplot(111)
            axis.text(0.5, 0.5, "未读取到有效光谱", ha="center", va="center")
            axis.set_axis_off()
        else:
            axes = self.figure.subplots(3, 3, squeeze=False)
            for axis, (spectrum_path, wavenumbers, intensities) in zip(axes.flat, spectra):
                line = axis.plot(
                    wavenumbers,
                    intensities,
                    color="#4C72B0",
                    linewidth=0.8,
                    picker=5,
                )[0]
                line.set_gid(str(spectrum_path))
                axis.set_title(spectrum_path.name, fontsize=9)
                axis.tick_params(labelsize=8)
                axis.grid(alpha=0.18)
            for axis in axes.flat[visible_count:]:
                axis.set_visible(False)

        self.figure.tight_layout(rect=(0, 0, 1, 0.95))
        self.write_completed_date_if_ready(date_dir, folder_dirs)
        self.figure.canvas.draw_idle()

    def write_completed_date_if_ready(self, date_dir: Path, folder_dirs: list[Path]) -> None:
        """在日期内所有光谱均显示过后写入审查日志。"""
        if date_dir.name in self.reviewed_date_names:
            return
        date_spectrum_paths = [
            spectrum_path
            for folder_dir in folder_dirs
            for spectrum_path in folder_dir.rglob("*.arc_data")
        ]
        if not all(path in self.reviewed_spectrum_paths for path in date_spectrum_paths):
            return
        write_reviewed_date(self.review_log_path, date_dir.name)
        self.reviewed_date_names.add(date_dir.name)
        print(f"已完成并记录日期：{date_dir.name}")

    def select_spectrum(self, event) -> None:
        """选中鼠标单击的光谱并突出显示对应曲线。"""
        if self.selected_line is not None:
            self.selected_line.set_color("#4C72B0")
            self.selected_line.set_alpha(0.32)
            self.selected_line.set_linewidth(0.75)

        self.selected_line = event.artist
        self.selected_path = Path(self.selected_line.get_gid())
        self.selected_line.set_color("#C44E52")
        self.selected_line.set_alpha(1.0)
        self.selected_line.set_linewidth(1.5)
        print(f"已选中：{self.selected_path}；按 Enter 删除该原文件。")
        self.figure.canvas.draw_idle()

    def delete_selected_spectrum(self) -> None:
        """删除选中的 .arc_data 原文件，并重绘当前子文件夹。"""
        if self.selected_path is None:
            print("请先单击一条光谱曲线，再按 Enter。")
            return
        if not self.selected_path.is_file():
            print(f"文件不存在，未执行删除：{self.selected_path}")
            self.draw_current_folder()
            return
        self.selected_path.unlink()
        write_deleted_spectrum(self.deleted_log_path, self.data_dir, self.selected_path)
        print(f"已删除原文件：{self.selected_path}")
        self.draw_current_folder()

    def advance_to_next_page_or_folder(self) -> None:
        """翻到下一页；当前子文件夹结束时依次进入下一个子文件夹或日期。"""
        date_dir = self.date_dirs[self.date_index]
        folder_dirs = self.date_folder_dirs[date_dir]
        folder_dir = folder_dirs[self.folder_index]
        spectrum_count = len(list(folder_dir.rglob("*.arc_data")))
        page_count = max(1, (spectrum_count + SPECTRA_PER_PAGE - 1) // SPECTRA_PER_PAGE)
        if self.page_index + 1 < page_count:
            self.page_index += 1
            return

        self.folder_index += 1
        self.page_index = 0
        if self.folder_index < len(folder_dirs):
            return
        self.date_index = (self.date_index + 1) % len(self.date_dirs)
        self.folder_index = 0

    def change_browse_position(self, event) -> None:
        """响应连续翻页，或以 Enter 删除选中的原文件。"""
        if event.key == "pagedown":
            self.advance_to_next_page_or_folder()
        elif event.key == "pageup":
            self.page_index -= 1
        elif event.key in {"enter", "return"}:
            self.delete_selected_spectrum()
            return
        else:
            return
        self.draw_current_folder()


def parse_arguments() -> argparse.Namespace:
    """读取待审查 cosmic_data 目录的命令行参数。"""
    parser = argparse.ArgumentParser(description="浏览并人工审查 cosmic_data 光谱")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR),
        help="cosmic_data 根目录；默认使用 dataset/alldata/cosmic_data",
    )
    return parser.parse_args()


def run_browser() -> None:
    """配置本地绘图窗口并启动 cosmic_data 浏览器。"""
    configure_chinese_font()
    arguments = parse_arguments()
    data_dir = resolve_data_dir(arguments.data_dir)
    browser = CosmicDataBrowser(data_dir)
    browser.show()


if __name__ == "__main__":
    run_browser()
