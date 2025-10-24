import os
import sys
from contextlib import redirect_stdout
from utils.dataset import CASIAwebfaceDataset
from utils.utils import Tee
from utils.criterion import SphereFaceNet
from utils.config import DATASET_PATH, WORKING_PATH
from utils.model_utils import main_pipeline
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Load dataset to get number of classes
    train_dataset = CASIAwebfaceDataset(
        root_dir=f'{DATASET_PATH}/CASIA-webface',
        transform=transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    )
    
    # Setup logging
    log_folder = f'{WORKING_PATH}/log'
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, 'sphereface.txt')
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main_pipeline(
                model_class=SphereFaceNet,
                model_name="SphereFace",
                project_name="sphereface-training",
                model_final_filename="sphereface_final.pth",
                model_best_filename="sphereface_best.pth",
                num_classes=train_dataset.num_of_identities,
                working_path=WORKING_PATH,
                dataset_path=DATASET_PATH
            )