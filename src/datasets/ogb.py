from typing import Optional
import torch

import torch_geometric.transforms as transforms
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch.utils.data import DataLoader
from torch_geometric.utils import to_undirected, add_self_loops
from pytorch_lightning import LightningDataModule

from src.datasets.NodeClassificationDataset import NodeClassificationDataset


def get_mask(idx, num_nodes):
    """
    Given a tensor of ids and a number of nodes, return a boolean mask of size num_nodes which is set to True at indices
    in `idx`, and to False for other indices.
    """
    mask = torch.zeros(num_nodes, dtype=torch.bool)
    mask[idx] = 1
    return mask


class OGBDataModule(LightningDataModule):
    def __init__(self,
                 name,
                 data_dir,
                 directed,
                 self_loops,
                 split_type,
                 split_index,
                 split_ratio,
                 seed,):
        super().__init__()
        self.name = name
        self.input_dim = None
        self.output_dim = None
        self.n_nodes = None
        self.data_dir = data_dir
        self.directed = directed
        self.self_loops = self_loops
        self.split_type = split_type
        self.split_index = split_index
        self.split_ratio = split_ratio
        self.seed = seed

    def prepare_data(self):
        # download data only
        pass

    def setup(self, stage: Optional[str] = None):
        if self.split_type == "random":
            raise NotImplementedError
        elif self.split_type == "public":
            dataset = PygNodePropPredDataset(name=self.name, transform=transforms.ToSparseTensor(), root=self.data_dir)
            evaluator = Evaluator(name=self.name)
            split_idx = dataset.get_idx_split()
            train_mask = get_mask(split_idx["train"], dataset.data.num_nodes)
            val_mask = get_mask(split_idx["valid"], dataset.data.num_nodes)
            test_mask = get_mask(split_idx["test"], dataset.data.num_nodes)
        else:
            raise NotImplementedError
        edge_index = dataset.data.edge_index
        if not self.directed:
            edge_index = to_undirected(edge_index)
        if self.self_loops:
            edge_index, _ = add_self_loops(edge_index)
        x = dataset.data.x
        y = dataset.data.y

        # assign to use in dataloaders methods
        self.train_dataset = NodeClassificationDataset(edge_index=edge_index, x=x, y=y, indices=train_mask, )
        self.val_dataset = NodeClassificationDataset(edge_index=edge_index, x=x, y=y, indices=val_mask, )
        self.test_dataset = NodeClassificationDataset(edge_index=edge_index, x=x, y=y, indices=test_mask, )
        self.n_nodes = x.shape[-2]
        self.input_dim = x.shape[-1]
        self.output_dim = int(y.max() + 1)

    def train_dataloader(self):
        return DataLoader(self.train_dataset)

    def val_dataloader(self):
        return DataLoader(self.val_dataset)

    def test_dataloader(self):
        return DataLoader(self.test_dataset)
