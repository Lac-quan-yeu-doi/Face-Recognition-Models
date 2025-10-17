import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

import os
from PIL import Image
from torch.utils.data import Dataset

class LFWDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Dataset structure:
        root/
            lfw_funneled/
                person1/
                    img1.jpg
                    img2.jpg
                person2/
                    img3.jpg
                    img4.jpg
            pairs.txt (optional)

        Args:
            root_dir (str): Path to dataset root.
            transform (callable, optional): torchvision transforms for preprocessing.
        """
        self.root_dir = root_dir
        self.identity_dir = os.path.join(root_dir, 'lfw_funneled')
        self.transform = transform

        # List all identities (folders)
        self.identities = [
            x for x in os.listdir(self.identity_dir)
            if os.path.isdir(os.path.join(self.identity_dir, x))
        ]
        self.idx_to_class = {i: name for i, name in enumerate(self.identities)}
        self.class_to_idx = {name: i for i, name in enumerate(self.identities)}
        self.num_of_identities = len(self.identities)

        # Prepare triplets
        self.anchors = []
        self.positives = []
        self.negatives = []

        # Collect all pair files (ignore pairs.txt)
        self.pair_files = [
            os.path.join(self.identity_dir, x)
            for x in os.listdir(self.identity_dir)
            if os.path.isfile(os.path.join(self.identity_dir, x)) and x != "pairs.txt"
        ]

        # Read pair files, construct triplets
        for pair_file in self.pair_files:
            with open(pair_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            for i in range(0, len(lines), 4):
                # Existence
                for j in range(4):
                    if not os.path.exists(os.path.join(self.identity_dir, lines[j])):
                        raise FileNotFoundError(f"{lines[j]} does not exist")

                # Append triplets (duplicated pattern kept as in your original)
                self.anchors.append(lines[i])
                self.anchors.append(lines[i])
                self.positives.append(lines[i + 1])
                self.positives.append(lines[i + 1])
                self.negatives.append(lines[i + 2])
                self.negatives.append(lines[i + 3])

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        # Get triplet paths
        anchor_path = os.path.join(self.identity_dir, self.anchors[index])
        positive_path = os.path.join(self.identity_dir, self.positives[index])
        negative_path = os.path.join(self.identity_dir, self.negatives[index])

        # Load images
        anchor = Image.open(anchor_path).convert("RGB")
        positive = Image.open(positive_path).convert("RGB")
        negative = Image.open(negative_path).convert("RGB")

        # Apply transforms
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


class CASIAwebfaceDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
        
