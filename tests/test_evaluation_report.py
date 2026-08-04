from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)

import matplotlib.image as image
import numpy as np

from raman.eval.report import save_confusion_matrix_figure
from ramanv2.evaluation.report import _write_confusion_matrix


def test_confusion_matrix_figure_matches_reference_layout(tmp_path: Path) -> None:
    matrix = np.array([[9, 1, 0], [0, 8, 2], [1, 0, 9]])
    class_names = ["GenusA/Prefix_1", "GenusB/Prefix_2", "GenusC/Prefix_3"]
    reference_path = tmp_path / "reference.png"
    output_path = tmp_path / "ramanv2.png"

    save_confusion_matrix_figure(matrix, class_names, reference_path)
    _write_confusion_matrix(output_path, matrix, class_names)

    np.testing.assert_array_equal(image.imread(output_path), image.imread(reference_path))
