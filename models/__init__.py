import os
from PIL import Image
from torch.utils.data import Dataset

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.data = []

        # Read annotation file
        with open(annotation_file, 'r') as f:
            for line in f.readlines():
                img_name, label = line.strip().split()
                img_path = os.path.join(root_dir, img_name)
                if os.path.exists(img_path):  # ✅ skip missing images
                    self.data.append((img_path, int(label)))

        if len(self.data) == 0:
            raise RuntimeError(f"No images found in {root_dir} with labels in {annotation_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
