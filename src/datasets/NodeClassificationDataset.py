from torch.utils.data import Dataset


class NodeClassificationDataset(Dataset):
    def __init__(self, edge_index, x, y, indices):
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.indices = indices

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        sample = {"edge_index": self.edge_index,
                  "x": self.x,
                  "y": self.y,
                  "indices": self.indices}
        return sample
