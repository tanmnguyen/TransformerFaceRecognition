import os 
import random 
from torch.utils.data import Dataset

random.seed(42)
class FaceDataset(Dataset):
    def __init__(self, img_path: str):
        super().__init__()
        ids = os.listdir(img_path)

        self.lbl, self.img_path = [], []
        for faceid in ids:
            files = os.listdir(os.path.join(img_path, faceid))
            for file in files:
                self.lbl.append(faceid)
                self.img_path.append(os.path.join(img_path, faceid, file))


        self.num_classes = len(set(self.lbl))

    def __len__(self):
        assert len(self.lbl) == len(self.img_path)
        return len(self.lbl)
    
    def __getitem__(self, idx):
        return self.img_path[idx], self.lbl[idx]
