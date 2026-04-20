"""PCA+SVM 基线评估入口"""

from raman.eval import BaselineOverrides, run_pca_svm_baseline

# 手动覆盖
EXP_DIR = ""
LEVEL = "level_1"
USE_ALL_CHANNELS = False
PCA_N_COMPONENTS = 10
SVM_C = 1.0
SVM_KERNEL = "rbf"
SVM_GAMMA = "scale"
RANDOM_STATE = 42


def main():
    overrides = BaselineOverrides(
        exp_dir=EXP_DIR,
        level=LEVEL,
        use_all_channels=USE_ALL_CHANNELS,
        pca_n_components=PCA_N_COMPONENTS,
        svm_c=SVM_C,
        svm_kernel=SVM_KERNEL,
        svm_gamma=SVM_GAMMA,
        random_state=RANDOM_STATE,
    )
    run_pca_svm_baseline(overrides)


if __name__ == "__main__":
    main()
