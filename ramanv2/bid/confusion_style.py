"""BID 论文风格的物种混淆矩阵。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from ramanv2.common.file_ops import read_json
from ramanv2.common.plotting import configure_matplotlib_fonts
from ramanv2.core.run_context import CONFUSION_MATRIX_FILE_NAME, METRICS_FILE_NAME


@dataclass(frozen=True)
class BidClassBand:
    """一组连续类别及其论文配色。"""

    name: str
    color: str
    classes: tuple[str, ...]


@dataclass(frozen=True)
class BidHighlightBlock:
    """论文中额外高亮的组内混淆块。"""

    name: str
    classes: tuple[str, ...]
    fill: str
    edge: str


@dataclass(frozen=True)
class BidConfusionStyle:
    """BID 混淆矩阵的布局、配色与标注规格。"""

    bands: tuple[BidClassBand, ...]
    blocks: tuple[BidHighlightBlock, ...]
    cell_background: str = "#FFFAE1"
    diagonal_fill: str = "#1E1A4A"
    figure_size: tuple[float, float] = (13.0, 13.5)
    normalization: str = "row"
    show_zero_enable: bool = False
    count_font_size: float = 6.4
    ratio_font_size: float = 5.2
    band_font_size: float = 8.0
    axis_title_font_size: float = 11.0


BID_CONFUSION_STYLE = BidConfusionStyle(
    bands=(
        BidClassBand(
            "Staphylococcus",
            "#D8F0F8",
            (
                "MRSA 1",
                "MRSA 2",
                "MSSA 1",
                "MSSA 2",
                "MSSA 3",
                "S. epidermidis",
                "S. lugdunensis",
            ),
        ),
        BidClassBand(
            "S. pneumoniae",
            "#C8D0F0",
            ("S. pneumoniae 1", "S. pneumoniae 2"),
        ),
        BidClassBand(
            "Streptococcus / E. faecalis",
            "#D8A8E8",
            (
                "Group A Strep",
                "Group B Strep",
                "Group C Strep",
                "Group G Strep",
                "S. sanguinis",
                "E. faecalis 1",
                "E. faecalis 2",
            ),
        ),
        BidClassBand("E. faecium", "#D078A8", ("E. faecium",)),
        BidClassBand(
            "Enterobacterales",
            "#B07058",
            (
                "E. coli 1",
                "E. coli 2",
                "K. pneumoniae 1",
                "K. pneumoniae 2",
                "K. aerogenes",
                "E. cloacae",
                "P. mirabilis",
                "S. marcescens",
            ),
        ),
        BidClassBand("S. enterica", "#787020", ("S. enterica",)),
        BidClassBand(
            "Pseudomonas",
            "#387020",
            ("P. aeruginosa 1", "P. aeruginosa 2"),
        ),
        BidClassBand(
            "Candida",
            "#102830",
            ("C. albicans", "C. glabrata"),
        ),
    ),
    blocks=(),
)


def write_bid_confusion_matrix(
    result_dir: Path | str,
    style: BidConfusionStyle = BID_CONFUSION_STYLE,
) -> Path:
    """按 BID 论文样式重绘结果目录中的混淆矩阵。"""
    target_dir = Path(result_dir)
    metrics = read_json(target_dir / METRICS_FILE_NAME)
    class_names = [str(item["name"]) for item in metrics["classes"]]
    matrix = np.asarray(metrics["matrix"], dtype=np.int64)
    order = [name for band in style.bands for name in band.classes]
    missing = [name for name in order if name not in class_names]
    extra = [name for name in class_names if name not in order]
    if missing or extra:
        raise ValueError(
            f"BID 样式类别与评估类别不一致：缺少 {missing}，多出 {extra}"
        )
    indices = [class_names.index(name) for name in order]
    ordered = matrix[np.ix_(indices, indices)]
    path = target_dir / CONFUSION_MATRIX_FILE_NAME
    _write_matrix_figure(path, ordered, order, style)
    return path


def _write_matrix_figure(
    path: Path,
    matrix: np.ndarray,
    class_names: list[str],
    style: BidConfusionStyle,
) -> None:
    """绘制类别分组的计数与比例混淆矩阵。"""
    configure_matplotlib_fonts()
    ratios = _normalized_matrix(matrix, style.normalization)
    band_ids, band_starts = _band_layout(class_names, style)
    block_ids, block_starts = _block_layout(class_names, style.blocks)
    count = len(class_names)
    left, bottom, width, height = 0.21, 0.055, 0.72, 0.72
    top_height = 0.18

    figure = plt.figure(figsize=style.figure_size)
    axis = figure.add_axes([left, bottom, width, height])
    top_axis = figure.add_axes([left, bottom + height, width, top_height])
    left_axis = figure.add_axes([left - 0.17, bottom, 0.16, height])

    for row in range(count):
        for col in range(count):
            facecolor = _cell_facecolor(
                matrix[row, col],
                float(ratios[row, col]),
                block_ids[row],
                block_ids[col],
                row == col,
                style,
            )
            axis.add_patch(
                Rectangle(
                    (col, row),
                    1,
                    1,
                    facecolor=facecolor,
                    edgecolor="#D9D4C2",
                    linewidth=0.25,
                )
            )
            _write_cell_text(
                axis,
                row,
                col,
                int(matrix[row, col]),
                float(ratios[row, col]),
                facecolor,
                style,
            )

    for start, _stop in band_starts:
        axis.axvline(start, color="#666666", linewidth=0.5, zorder=3)
        axis.axhline(start, color="#666666", linewidth=0.5, zorder=3)
        top_axis.axvline(start, color="white", linewidth=0.8, zorder=3)
        left_axis.axhline(start, color="white", linewidth=0.8, zorder=3)
    for (start, stop), block in zip(block_starts, style.blocks):
        size = stop - start
        axis.add_patch(
            Rectangle(
                (start, start),
                size,
                size,
                fill=False,
                edgecolor=block.edge,
                linewidth=1.1,
                zorder=3,
            )
        )
    axis.axvline(count, color="#666666", linewidth=0.5, zorder=3)
    axis.axhline(count, color="#666666", linewidth=0.5, zorder=3)
    axis.add_patch(
        Rectangle(
            (0, 0),
            count,
            count,
            fill=False,
            edgecolor="black",
            linewidth=1.2,
            zorder=4,
        )
    )

    _draw_top_bands(top_axis, class_names, band_ids, style)
    _draw_left_bands(left_axis, class_names, band_ids, style)
    axis.set_xlim(0, count)
    axis.set_ylim(count, 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.text(
        left + width / 2,
        bottom + height + top_height + 0.008,
        "Predicted",
        ha="center",
        va="bottom",
        fontsize=style.axis_title_font_size,
        fontweight="bold",
    )
    figure.text(
        0.02,
        bottom + height / 2,
        "True",
        ha="center",
        va="center",
        rotation=90,
        fontsize=style.axis_title_font_size,
        fontweight="bold",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300)
    plt.close(figure)


def _normalized_matrix(matrix: np.ndarray, normalization: str) -> np.ndarray:
    """按行、列或总数归一化；零分母按 1 处理。"""
    if normalization == "row":
        divisor = matrix.sum(axis=1, keepdims=True)
    elif normalization == "col":
        divisor = matrix.sum(axis=0, keepdims=True)
    elif normalization == "total":
        divisor = np.asarray([[matrix.sum()]])
    else:
        raise ValueError(f"未知归一化方式：{normalization}")
    divisor = np.where(divisor == 0, 1, divisor)
    return matrix.astype(np.float64) / divisor


def _band_layout(
    class_names: list[str],
    style: BidConfusionStyle,
) -> tuple[list[int], list[tuple[int, int]]]:
    """返回每个类别的组索引与各组连续区间。"""
    band_ids: list[int] = []
    band_starts: list[tuple[int, int]] = []
    cursor = 0
    for band_id, band in enumerate(style.bands):
        band_ids.extend([band_id] * len(band.classes))
        band_starts.append((cursor, cursor + len(band.classes)))
        cursor += len(band.classes)
    return band_ids, band_starts


def _block_layout(
    class_names: list[str],
    blocks: tuple[BidHighlightBlock, ...],
) -> tuple[list[int], list[tuple[int, int]]]:
    """返回每个类别的论文高亮块索引与各块连续区间。"""
    class_positions = {name: index for index, name in enumerate(class_names)}
    block_ids = [-1] * len(class_names)
    block_starts: list[tuple[int, int]] = []
    for block_id, block in enumerate(blocks):
        positions = [class_positions[name] for name in block.classes]
        start, stop = min(positions), max(positions) + 1
        if stop - start != len(positions):
            raise ValueError(f"高亮块类别不连续：{block.name}")
        block_starts.append((start, stop))
        for position in positions:
            block_ids[position] = block_id
    return block_ids, block_starts


def _cell_facecolor(
    value: int,
    ratio: float,
    row_block: int,
    col_block: int,
    is_diagonal: bool,
    style: BidConfusionStyle,
) -> str:
    """论文高亮块使用浅色底，对角线按 recall 做深浅渐变。"""
    if is_diagonal and value > 0:
        return _blend_with_white(style.diagonal_fill, 0.35 + 0.65 * min(max(ratio, 0.0), 1.0))
    if row_block >= 0 and row_block == col_block:
        return style.blocks[row_block].fill
    return style.cell_background


def _write_cell_text(
    axis,
    row: int,
    col: int,
    count: int,
    ratio: float,
    facecolor: str,
    style: BidConfusionStyle,
) -> None:
    """在格内写入计数与比例；零值默认留空。"""
    if count == 0 and not style.show_zero_enable:
        return
    color = _text_color(facecolor)
    axis.text(
        col + 0.5,
        row + 0.40,
        str(count),
        ha="center",
        va="center",
        fontsize=style.count_font_size,
        color=color,
    )
    axis.text(
        col + 0.5,
        row + 0.69,
        f"{ratio * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=style.ratio_font_size,
        color=color,
    )


def _draw_top_bands(
    axis,
    class_names: list[str],
    band_ids: list[int],
    style: BidConfusionStyle,
) -> None:
    """绘制顶部彩色类别条与旋转标签。"""
    axis.set_xlim(0, len(class_names))
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    for index, name in enumerate(class_names):
        color = style.bands[band_ids[index]].color
        axis.add_patch(
            Rectangle(
                (index, 0),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
        )
        axis.text(
            index + 0.5,
            0.06,
            _display_name(name),
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=style.band_font_size,
            color=_text_color(color),
        )


def _draw_left_bands(
    axis,
    class_names: list[str],
    band_ids: list[int],
    style: BidConfusionStyle,
) -> None:
    """绘制左侧彩色类别条与横排标签。"""
    axis.set_xlim(0, 1)
    axis.set_ylim(len(class_names), 0)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    for index, name in enumerate(class_names):
        color = style.bands[band_ids[index]].color
        axis.add_patch(
            Rectangle(
                (0, index),
                1,
                1,
                facecolor=color,
                edgecolor="white",
                linewidth=0.5,
            )
        )
        axis.text(
            0.94,
            index + 0.5,
            _display_name(name),
            ha="right",
            va="center",
            fontsize=style.band_font_size,
            color=_text_color(color),
        )


def _display_name(name: str) -> str:
    """补全论文中的 Strep. 缩写点号。"""
    if name.startswith("Group ") and name.endswith(" Strep"):
        return f"{name}."
    return name


def _text_color(color: str) -> str:
    """按底色亮度选择黑字或白字。"""
    red, green, blue = _hex_to_rgb(color)
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    return "white" if luminance < 140 else "black"


def _blend_with_white(color: str, ratio: float) -> str:
    """按比例把颜色混入白色，ratio 为保留颜色的比例。"""
    red, green, blue = _hex_to_rgb(color)
    values = [round(channel * ratio + 255 * (1 - ratio)) for channel in (red, green, blue)]
    return "#{:02X}{:02X}{:02X}".format(*values)


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    """把 #RRGGBB 转为 RGB 三元组。"""
    text = color.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
