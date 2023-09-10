# Code modified from https://github.com/dtuggener/LEDGAR_provision_classification and https://github.com/vsingh-group/LCODEC-deep-unlearning

import copy
import numpy as np
import torch.utils.data
from torch.utils.data import Subset


class LabelSubsetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, which_labels=(0, 1)):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        :param which_labels: which labels to use
        """
        super(LabelSubsetWrapper, self).__init__()
        self.dataset = dataset
        self.which_labels = which_labels
        # record important attributes
        if hasattr(dataset, 'statistics'):
            self.statistics = dataset.statistics
        self.valid_indices = [idx for idx, (x, y) in enumerate(dataset) if y in which_labels]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.valid_indices[idx]]
        assert y in self.which_labels
        new_y = self.which_labels.index(y)
        return x, torch.tensor(new_y, dtype=torch.long)


BinaryDatasetWrapper = LabelSubsetWrapper  # shortcut


class SubsetDataWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, exclude_indices=None, include_indices=None):
        """
        :param dataset: StandardVisionDataset or derivative class instance
        """
        super(SubsetDataWrapper, self).__init__()

        if exclude_indices is None:
            assert include_indices is not None
        if include_indices is None:
            assert exclude_indices is not None

        self.dataset = dataset

        if include_indices is not None:
            self.include_indices = include_indices
        else:
            S = set(exclude_indices)
            self.include_indices = [idx for idx in range(len(dataset)) if idx not in S]

        # record important attributes
        if hasattr(dataset, 'statistics'):
            self.statistics = dataset.statistics

    def __len__(self):
        return len(self.include_indices)

    def __getitem__(self, idx):
        real_idx = self.include_indices[idx]
        return self.dataset[real_idx]

