import os
import csv
import random
from PIL import Image
from torch.utils.data import Dataset
from alive_progress import alive_bar

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
        if not os.path.exists(self.identity_dir):
            raise FileNotFoundError(f"Directory {self.identity_dir} does not exist")
        self.transform = transform

        # List all identities (folders)
        self.samples = []
        self.identities = [
            x for x in os.listdir(self.identity_dir)
            if os.path.isdir(os.path.join(self.identity_dir, x))
        ]
        self.idx_to_class = {i: name for i, name in enumerate(self.identities)}
        self.class_to_idx = {name: i for i, name in enumerate(self.identities)}
        self.num_of_identities = len(self.identities)

        # Create (image_path, label) pairs
        with alive_bar(len(self.identities), title='Loading LFW dataset', bar='filling', spinner='waves', length=20) as bar:
            for identity in self.identities:
                identity_path = os.path.join(self.identity_dir, identity)
                label = self.class_to_idx[identity]
                
                for image in os.listdir(identity_path):
                    if image.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity, image)
                        self.samples.append((img_path, label))
                        bar()

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image_path = os.path.join(self.identity_dir, image_path)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        if self.transform:
            image = self.transform(image)

        return image, label

class CASIAwebfaceDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Dataset structure:
        root/
            casia-webface/
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
        self.identity_dir = os.path.join(root_dir, split)
        if not os.path.exists(self.identity_dir):
            raise FileNotFoundError(f"Directory {self.identity_dir} does not exist")
        self.transform = transform

        # List all identities (folders)
        self.samples = []
        self.identities = [
            x for x in os.listdir(self.identity_dir)
            if os.path.isdir(os.path.join(self.identity_dir, x))
        ]
        self.idx_to_class = {i: name for i, name in enumerate(self.identities)}
        self.class_to_idx = {name: i for i, name in enumerate(self.identities)}
        self.num_of_identities = len(self.identities)

        # Create (image_path, label) pairs
        with alive_bar(len(self.identities), title='Loading CASIAwebface dataset', bar='smooth', spinner='dots_waves', length=20) as bar:
            for identity in self.identities:
                identity_path = os.path.join(self.identity_dir, identity)
                label = self.class_to_idx[identity]
                for image in os.listdir(identity_path):
                    if image.lower().endswith(('.jpg', '.jpeg', '.png')):
                        img_path = os.path.join(identity, image)
                        self.samples.append((img_path, label)) 
                        bar()
        
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image_path = os.path.join(self.identity_dir, image_path)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
        
        if self.transform:
            image = self.transform(image)

        return image, label

# # Old LFWDataset class for validation
# class LFWPairDataset(Dataset):
#     def __init__(self, root_dir, pairs_file, transform=None):
#         """
#         Dataset for LFW pairwise verification using pairs.txt.
#         Args:
#             root_dir (str): Path to dataset root.
#             pairs_file (str): Path to pairs.txt file.
#             transform (callable, optional): torchvision transforms for preprocessing.
#         """
#         self.root_dir = root_dir
#         self.identity_dir = os.path.join(root_dir, 'lfw_funneled')
#         if not os.path.exists(self.identity_dir):
#             raise FileNotFoundError(f"Directory {self.identity_dir} does not exist")
#         self.transform = transform
#         self.pairs = []
        
#         # Parse pairs.txt (format: same identity: name img1 img2; different: name1 img1 name2 img2)
#         with open(pairs_file, 'r') as f:
#             lines = f.readlines()[1:]  # Skip header
#             for line in lines:
#                 parts = line.strip().split()
#                 if len(parts) == 3:  # Same identity
#                     name, img1, img2 = parts
#                     self.pairs.append((f"{name}/{name}_{img1.zfill(4)}.jpg", 
#                                      f"{name}/{name}_{img2.zfill(4)}.jpg", 1))
#                 elif len(parts) == 4:  # Different identities
#                     name1, img1, name2, img2 = parts
#                     self.pairs.append((f"{name1}/{name1}_{img1.zfill(4)}.jpg", 
#                                      f"{name2}/{name2}_{img2.zfill(4)}.jpg", 0))
    
#     def __len__(self):
#         return len(self.pairs)
    
#     def __getitem__(self, index):
#         img1_path, img2_path, same = self.pairs[index]
#         img1_path = os.path.join(self.identity_dir, img1_path)
#         img2_path = os.path.join(self.identity_dir, img2_path)
#         try:
#             img1 = Image.open(img1_path).convert('RGB')
#             img2 = Image.open(img2_path).convert('RGB')
#         except Exception as e:
#             print(f"Error loading images {img1_path} or {img2_path}: {e}")
#             return None
        
#         if self.transform:
#             img1 = self.transform(img1)
#             img2 = self.transform(img2)
        
#         return img1, img2, same

# New LWFDataset class for a Phuong class
class LFWPairDataset(Dataset):
    def __init__(self, root_dir, pairs_files, transform=None):
        """
        Dataset for LFW pairwise verification using CSV pair files (e.g., pairs.csv, matchpairs*.csv, mismatchpairs*.csv).
        Args:
            root_dir (str): Path to dataset root containing aligned images (e.g., aligned_lfw_path).
            pairs_files (str or list): Path to a single CSV file (e.g., pairs.csv) or list of CSV files 
                                       (e.g., [matchpairsDevTrain.csv, mismatchpairsDevTrain.csv]).
            transform (callable, optional): torchvision transforms for preprocessing.
        """
        self.root_dir = root_dir
        self.identity_dir = os.path.join(self.root_dir, "lfw-deepfunneled")
        if not os.path.exists(self.identity_dir):
            raise FileNotFoundError(f"Directory {self.identity_dir} does not exist")
        self.transform = transform
        self.pairs = []

        # Handle single file or list of files
        if isinstance(pairs_files, str):
            pairs_files = [pairs_files]
        # elif not isinstance(pairs_files, list):
        #     raise ValueError("pairs_files must be a string or list of strings")

        # Parse CSV files
        if pairs_files is not None:
            for pairs_file in pairs_files:
                with open(pairs_file, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)  # Skip header (e.g., name,imagenum1,imagenum2,)
                    if header is None:
                        raise ValueError(f"Empty or invalid CSV file: {pairs_file}")
                    
                    for row in reader:
                        if len(row) == 3:  # Matched pair: name,imagenum1,imagenum2
                            name, img1, img2 = row
                            img1_path = f"{name}/{name}_{img1.zfill(4)}.jpg"
                            img2_path = f"{name}/{name}_{img2.zfill(4)}.jpg"
                            self.pairs.append((img1_path, img2_path, 1))
                        elif len(row) == 4:  # Mismatched pair: name1,imagenum1,name2,imagenum2
                            name1, img1, name2, img2 = row
                            img1_path = f"{name1}/{name1}_{img1.zfill(4)}.jpg"
                            img2_path = f"{name2}/{name2}_{img2.zfill(4)}.jpg"
                            self.pairs.append((img1_path, img2_path, 0))
                        else:
                            print(f"Skipping invalid row in {pairs_file}: {row}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img1_path, img2_path, same = self.pairs[index]
        img1_path = os.path.join(self.identity_dir, img1_path)
        img2_path = os.path.join(self.identity_dir, img2_path)
        
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
        except Exception as e:
            print(f"Error loading images {img1_path} or {img2_path}: {e}")
            return None

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, same
