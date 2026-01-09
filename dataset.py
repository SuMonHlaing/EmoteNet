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
        # We read lines separately to get the total count for the print statement
        with open(annotation_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue # Skip empty or malformed lines
            
            img_name, label_str = parts[0], parts[1]
            
            # Adjust filename for aligned images
            aligned_name = img_name.replace(".jpg", "_aligned.jpg")

            img_path = os.path.join(root_dir, aligned_name)
            
            if os.path.isfile(img_path):
                # *** FIX APPLIED HERE ***
                # Convert 1-indexed labels (1 to 7) to 0-indexed labels (0 to 6) 
                # for PyTorch's CrossEntropyLoss.
                label_0_indexed = int(label_str) - 1
                self.data.append((img_path, label_0_indexed))

        print(f"✅ Found {len(self.data)} existing images out of {len(lines)}")

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