"""
Handling vertically partitioned data
"""
from copy import deepcopy
from typing import List
from typing import Tuple
from typing import TypeVar
from uuid import uuid4

from PIL import Image
import numpy as np

# Ignore errors when running mypy script
# mypy: ignore-errors

Dataset = TypeVar("Dataset")


def add_ids(cls):
    """Decorator to add unique IDs to a dataset

    Args:
        cls (torch.utils.data.Dataset) : dataset to generate IDs for

    Returns:
        VerticalDataset : A class which wraps cls to add unique IDs as an attribute,
            and returns data, target, id when __getitem__ is called
    """

    class VerticalDataset(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.ids = np.array([uuid4() for _ in range(len(self))])

        def __getitem__(self, index):
            if self.data is None:
                img = None
            else:
                img = self.data[index]
                img = Image.fromarray(img.numpy(), mode="L")

                if self.transform is not None:
                    img = self.transform(img)

            if self.targets is None:
                target = None
            else:
                target = int(self.targets[index]) if self.targets is not None else None

                if self.target_transform is not None:
                    target = self.target_transform(target)

            id = self.ids[index]

            # Return a tuple of non-None elements
            return (*filter(lambda x: x is not None, (img, target, id)),)

        def __len__(self):
            if self.data is not None:
                return self.data.size(0)
            else:
                return len(self.targets)

        def get_ids(self) -> List[str]:
            """Return a list of the ids of this dataset."""
            return [str(id_) for id_ in self.ids]

        def sort_by_ids(self):
            """
            Sort the dataset by IDs in ascending order
            """
            ids = self.get_ids()
            sorted_idxs = np.argsort(ids)

            if self.data is not None:
                self.data = self.data[sorted_idxs]

            if self.targets is not None:
                self.targets = self.targets[sorted_idxs]

            self.ids = self.ids[sorted_idxs]

    return VerticalDataset


def partition_dataset(
    dataset: Dataset,
    keep_order: bool = False,
    remove_data: bool = True,
    n_of_partition: int = 1
) -> Tuple[Dataset, Dataset]:
    """Vertically partition a torch dataset in N (default=2)

    A vertical partition is when parameters for a single data point is
    split across multiple data holders.
    This function assumes the dataset to split contains images (e.g. MNIST).
    One dataset gets the images, the other gets the labels

    Args:
        dataset (torch.utils.data.Dataset) : The dataset to split. Must be a dataset of images, containing ids
        keep_order (bool, default = False) : If False, shuffle the elements of each dataset
        remove_data (bool, default = True) : If True, remove datapoints with probability 0.01

    Returns:
        torch.utils.data.Dataset : Dataset containing the first partition: the data/images
        torch.utils.data.Dataset : Dataset containing the second partition: the labels

    Raises:
        RuntimeError : If dataset does not have an 'ids' attribute
        AssertionError : If the size of the provided dataset
            does not have three elements (i.e. is not an image dataset)
    """
    if not hasattr(dataset, "ids"):
        raise RuntimeError("Dataset does not have attribute 'ids'")

    label_partition = deepcopy(dataset)
    data_partitions = []
    for i in range(n_of_partition):
        data_partitions.append(deepcopy(dataset))

    # Partition data
    label_partition.data = None
    for dp in data_partitions:
        dp.targets = None

    # Re-index data
    idxlbl = np.arange(len(label_partition))
    idxdt = [np.arange(len(dp)) for dp in data_partitions]

    # Remove random subsets of data with 1% prob
    if remove_data:
        idxlbl = np.random.uniform(0, 1, len(idxlbl)) > 0.01
        for i in range(len(idxdt)):
            idxdt[i] = np.random.uniform(0, 1, len(data_partitions[i])) > 0.01

    if not keep_order:
        np.random.shuffle(idxdt)
        for idx in idxdt:
            np.random.shuffle(idx)

    label_partition.targets = label_partition.targets[idxlbl]
    label_partition.ids = label_partition.ids[idxlbl]

    for i in range(len(data_partitions)):
        data_partitions[i].data = data_partitions[i].data[idxdt[i]]
        data_partitions[i].ids = data_partitions[i].ids[idxdt[i]]

    return label_partition, data_partitions
