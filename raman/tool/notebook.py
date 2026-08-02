"""供训练 notebook 复用的结果查看与打包辅助工具。"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from raman.analysis import HeatmapConfig
from raman.experiment import select_run_dir


@dataclass
class NotebookTools:
    """集中保存 notebook 的少量展示与分析默认值。"""

    baseline_use_all_channels: bool = False
    baseline_pca_n_components: float = 0.95
    baseline_svm_c: float = 1.0
    baseline_svm_kernel: str = "rbf"
    baseline_svm_gamma: str = "scale"
    baseline_random_state: int = 42
    inherit_missing_levels: bool = False
    heatmap_num_batches: int = 10
    heatmap_steps: int = 32
    heatmap_max_per_class: int = 50
    heatmap_row_norm: str = "max"
    heatmap_use_train_loader: bool = True
    heatmap_separate_class_plots: bool = True
    analysis_show_max_images: int = 12
    confusion_matrix_width: int | None = None
    confusion_matrix_height: int | None = None

    def baseline_kwargs(self):
        """返回 baseline 计算参数。"""
        return {
            "use_all_channels": self.baseline_use_all_channels,
            "pca_n_components": self.baseline_pca_n_components,
            "svm_c": self.baseline_svm_c,
            "svm_kernel": self.baseline_svm_kernel,
            "svm_gamma": self.baseline_svm_gamma,
            "random_state": self.baseline_random_state,
        }

    def heatmap_config(self):
        """返回 IG 热图参数。"""
        return HeatmapConfig(
            num_batches=self.heatmap_num_batches,
            steps=self.heatmap_steps,
            max_per_class=self.heatmap_max_per_class,
            row_norm=self.heatmap_row_norm,
            use_train_loader=self.heatmap_use_train_loader,
            separate_class_plots=self.heatmap_separate_class_plots,
        )

    @staticmethod
    def clear_run_selection():
        """清除临时 run 选择环境变量。"""
        os.environ.pop("RAMAN_RUN_SELECTION", None)

    @staticmethod
    def apply_run_selection(run_selection):
        """设置临时 run 选择环境变量。"""
        if run_selection:
            os.environ["RAMAN_RUN_SELECTION"] = json.dumps(
                run_selection,
                ensure_ascii=False,
            )
            print("RAMAN_RUN_SELECTION =", os.environ["RAMAN_RUN_SELECTION"])
        else:
            NotebookTools.clear_run_selection()
            print("RAMAN_RUN_SELECTION cleared")

    @staticmethod
    def _has_run(slot_dir):
        return any(path.is_dir() and path.name.startswith("run_") for path in slot_dir.iterdir())

    @classmethod
    def _model_slot_dirs(cls, exp_dir):
        root = Path(exp_dir)
        if not root.exists():
            return []
        slots = []
        for level_dir in sorted(root.glob("level_*")):
            if not level_dir.is_dir():
                continue
            if cls._has_run(level_dir):
                slots.append(level_dir)
            for child_dir in sorted(level_dir.iterdir()):
                if child_dir.is_dir() and child_dir.name.startswith(level_dir.name + "_"):
                    if cls._has_run(child_dir):
                        slots.append(child_dir)
        return slots

    @classmethod
    def list_run_slots(cls, exp_dir):
        """打印实验根内已有的层级或父类模型 run。"""
        root = Path(exp_dir)
        print("\n[Run 列表]")
        for slot_dir in cls._model_slot_dirs(root):
            key = slot_dir.relative_to(root).as_posix()
            runs = sorted(
                path.name
                for path in slot_dir.iterdir()
                if path.is_dir() and path.name.startswith("run_")
            )
            print(f"  {key}")
            print(f"    runs: {runs}")

    @staticmethod
    def auto_single_run_dir(exp_dir, level=None, parent_idx=None):
        """按实验根、层级和父类位置选择最新的单模型 run。"""
        if exp_dir is None or level is None:
            return None
        slot_dir = Path(exp_dir) / level
        if parent_idx is not None:
            slot_dir = slot_dir / f"{level}_{int(parent_idx)}"
        run_dir, _ = select_run_dir(slot_dir)
        return str(run_dir) if run_dir is not None else None

    def show_confusion_matrix(self, result_dir, filename="confusion_matrix.png", width=None, height=None):
        """显示指定结果目录中的混淆矩阵。"""
        from IPython.display import Image, display

        cm_path = Path(result_dir) / filename
        if not cm_path.exists():
            raise FileNotFoundError(f"找不到混淆矩阵：{cm_path}")
        width = self.confusion_matrix_width if width is None else width
        height = self.confusion_matrix_height if height is None else height
        image_kwargs = {}
        if width is not None:
            image_kwargs["width"] = width
        if height is not None:
            image_kwargs["height"] = height
        display(Image(filename=str(cm_path), **image_kwargs))
        print("confusion_matrix =", cm_path)

    def show_analysis_figures(self, result_dir, max_images=None):
        """依次显示 analysis 产出的前若干张图。"""
        from IPython.display import Image, display

        fig_dir = Path(result_dir) / "figures"
        if not fig_dir.exists():
            raise FileNotFoundError(f"找不到 analysis figures 目录：{fig_dir}")
        pngs = sorted(fig_dir.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"analysis figures 目录下没有 png：{fig_dir}")
        max_images = self.analysis_show_max_images if max_images is None else max_images
        for image_path in pngs[:max_images]:
            print(image_path.name)
            display(Image(filename=str(image_path)))
        if len(pngs) > max_images:
            print(f"还有 {len(pngs) - max_images} 张图未展示，目录：{fig_dir}")

    @staticmethod
    def package_directory(source_dir, output_dir="/content", package_name=None):
        """将实验目录完整压缩为 zip。"""
        source_dir = Path(source_dir)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"找不到要压缩的目录：{source_dir}")
        package_name = package_name or source_dir.name
        zip_path = shutil.make_archive(
            base_name=str(Path(output_dir) / package_name),
            format="zip",
            root_dir=str(source_dir.parent),
            base_dir=source_dir.name,
        )
        print("package_source =", source_dir)
        print("package_zip =", zip_path)
        return zip_path
