import random
from collections import defaultdict

from torch.utils.data import Sampler, Subset


class AutoHierarchicalBatchSampler(Sampler):
    """
    分层采样器。

    目标是在一个 batch 中尽量同时覆盖多个顶层类别和多个 leaf，
    从而让分析或调试时的 batch 结构更丰富。
    """

    def __init__(
        self,
        dataset,
        batch_size,
        top_level="level_1",
        leaf_level="leaf",
        min_samples_per_leaf=2,
        seed=42,
        shuffle=True,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.min_k = int(min_samples_per_leaf)
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        if isinstance(dataset, Subset):
            self.base_dataset = dataset.dataset
            self.subset_indices = list(dataset.indices)
        else:
            self.base_dataset = dataset
            self.subset_indices = None

        self.hier_names = self.base_dataset.hier_names
        self.label_maps = {
            name: self.base_dataset.label_maps_by_level[i]
            for i, name in enumerate(self.base_dataset.head_names)
        }

        self.top_level = self._resolve_level_name(top_level)
        self.leaf_level = self._resolve_level_name(leaf_level)
        self.top_to_leaf = self._build_index()
        self.num_samples = len(self.dataset)

    def _resolve_level_name(self, level_name):
        """
        兼容 `Dataset` 自带的层级名解析逻辑。
        """
        if hasattr(self.base_dataset, "_resolve_level_name"):
            return self.base_dataset._resolve_level_name(level_name)
        return level_name

    def _to_base_index(self, local_idx):
        """
        把 `Subset` 中的局部索引还原为原始数据集索引。
        """
        if self.subset_indices is None:
            return local_idx
        return self.subset_indices[local_idx]

    def _build_index(self):
        """
        建立 `top_id -> leaf_id -> [local_indices]` 的采样索引。
        """
        index = defaultdict(lambda: defaultdict(list))

        for local_idx in range(len(self.dataset)):
            base_idx = self._to_base_index(local_idx)
            hier = self.hier_names[base_idx]

            top_name = hier.get(self.top_level)
            leaf_name = hier.get(self.leaf_level)
            if top_name is None or leaf_name is None:
                continue

            top_id = self.label_maps[self.top_level][top_name]
            leaf_id = self.label_maps[self.leaf_level][leaf_name]
            index[top_id][leaf_id].append(local_idx)

        return index

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        for top_dict in self.top_to_leaf.values():
            for sample_indices in top_dict.values():
                self.rng.shuffle(sample_indices)

        batches = []
        while len(batches) * self.batch_size < self.num_samples:
            batch = self._sample_one_batch()
            if len(batch) < self.batch_size:
                break
            batches.append(batch)

        if self.shuffle:
            self.rng.shuffle(batches)

        for batch in batches:
            yield batch

    def _sample_one_batch(self):
        """
        生成一个尽量覆盖多个 top-level 和多个 leaf 的 batch。
        """
        batch = []
        top_ids_all = list(self.top_to_leaf.keys())
        self.rng.shuffle(top_ids_all)

        num_top = min(len(top_ids_all), max(2, self.batch_size // 8))
        top_ids = top_ids_all[:num_top]
        per_top_budget = self.batch_size // max(1, num_top)

        for top_id in top_ids:
            if len(batch) >= self.batch_size:
                break

            leaf_dict = self.top_to_leaf[top_id]
            leaf_ids = list(leaf_dict.keys())
            self.rng.shuffle(leaf_ids)

            num_leaf = min(
                len(leaf_ids),
                max(1, per_top_budget // max(1, self.min_k)),
            )
            leaf_ids = leaf_ids[:num_leaf]

            for leaf_id in leaf_ids:
                if len(batch) >= self.batch_size:
                    break

                indices = leaf_dict[leaf_id]
                if not indices:
                    continue

                remaining = self.batch_size - len(batch)
                sample_count = min(
                    len(indices),
                    max(self.min_k, remaining // max(1, (num_top * num_leaf))),
                )
                batch.extend(self.rng.sample(indices, sample_count))

        if len(batch) < self.batch_size:
            remain = self.batch_size - len(batch)
            all_indices = []
            candidates = []

            for top_dict in self.top_to_leaf.values():
                for indices in top_dict.values():
                    all_indices.extend(indices)
                    if len(indices) >= 2:
                        candidates.extend(indices)

            if len(candidates) >= remain:
                batch.extend(self.rng.sample(candidates, remain))
            else:
                batch.extend(self.rng.sample(all_indices, remain))

        return batch
