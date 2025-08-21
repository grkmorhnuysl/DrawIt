from pathlib import Path
import numpy as np, torch
from torch.utils.data import Dataset

class QuickDrawNPY(Dataset):
    def __init__(self, data_dir, class_names, per_class_limit=None, transform=None):
        self.samples, self.labels, self.class_names = [], [], class_names
        self.transform = transform
        data_dir = Path(data_dir)/'bitmap'
        for ci, name in enumerate(class_names):
            f = data_dir/f"{name}.npy"
            arr = np.load(f, mmap_mode='r')
            if per_class_limit: arr = arr[:per_class_limit]
            self.samples.append(arr)
            self.labels += [ci]*len(arr)
        self.samples = np.concatenate([s for s in self.samples], axis=0).astype('float32')/255.0
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        img = self.samples[idx][None, ...]
        y = self.labels[idx]
        if self.transform: img = self.transform(img)
        return torch.from_numpy(img), torch.tensor(y).long()
