#!/usr/bin/env python3

from torch.utils.data import Dataset

class ChatDataset(Dataset):
    """
    This class helps getting the chat dataset info
    """
    
    def __init__(self, X_train, y_train):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples