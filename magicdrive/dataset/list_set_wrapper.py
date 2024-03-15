import torch


class ListSetWrapper(torch.utils.data.DataLoader):
    def __init__(self, dataset, list) -> None:
        self.dataset = dataset
        self.list = list

    def __getitem__(self, idx):
        return self.dataset[self.list[idx]]

    def __len__(self):
        return len(self.list)
