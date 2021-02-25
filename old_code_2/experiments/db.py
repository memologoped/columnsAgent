import torch
from torch.utils.data import Dataset


class RecoveryHistory(Dataset):
    def __init__(self, max_history_size: int):
        self.max_history_size = max_history_size
        self.count = self.max_history_size
        self.history = list()
        self.temp_history = list()
        self.training = False

    def put(self, env_state, action, reward) -> None:
        self.count -= 1

    def reset(self) -> None:
        self.count = self.max_history_size

    def train_mode(self) -> None:
        if not self.training:
            self.temp_history = self.history.copy()

    def gather_mode(self) -> None:
        if self.training:
            self.history = self.temp_history

    def is_full(self) -> bool:
        return self.count <= 0

    def __getitem__(self, index: int) -> (torch.tensor, torch.tensor, torch.tensor):
        pass  # return tensors ([observation, action], reward)

    def __len__(self) -> int:
        return len(self.history)
