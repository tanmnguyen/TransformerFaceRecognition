import os 
import random 
from torch.utils.data import Dataset

random.seed(42)
class TripleFaceDataset(Dataset):
    def __init__(self, img_path: str):
        super().__init__()

        ids = os.listdir(img_path)
        
        # remove hidden files
        ids = [i for i in ids if not i.startswith(".")]

        self.triplets = [] 
        for faceid in ids:
           files = os.listdir(os.path.join(img_path, faceid))
           for i in range(len(files)):
                num_triplets = min(1, len(files))

                # randomly sample num_triplets
                for _ in range(num_triplets):
                    img_path1 = os.path.join(img_path, faceid, files[i]) 
                    img_path2 = img_path1
                    while img_path2 == img_path1 and len(files) > 1:
                        img_path2 = os.path.join(img_path, faceid, files[random.randint(0, len(files) - 1)])

                    # pick a random image from another face
                    random_faceid = faceid
                    while random_faceid == faceid:
                        random_faceid = ids[random.randint(0, len(ids) - 1)]
                        
                    random_files = os.listdir(os.path.join(img_path, random_faceid))
                    img_path3 = os.path.join(img_path, random_faceid, random_files[random.randint(0, len(random_files) - 1)])

                    self.triplets.append((img_path1, img_path2, img_path3))

    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]
