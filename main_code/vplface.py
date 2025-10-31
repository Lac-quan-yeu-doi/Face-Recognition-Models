# ver 2

import os
import sys
from contextlib import redirect_stdout
from utils.utils import Tee
from utils.criterion import VPLArcFaceNet
from utils.config import DATASET_PATH, WORKING_PATH
from utils.model_utils import main_pipeline
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Setup logging
    log_folder = f'{WORKING_PATH}/log'
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, 'vpl_arcface.txt')
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main_pipeline(
                model_class=VPLArcFaceNet,
                model_name="VPLArcFace",
                project_name="vpl_arcface-training",
                model_final_filename="vpl_arcface_final.pth",
                model_best_filename="vpl_arcface_best.pth",
                num_classes=10575, # 10575 for CASIA-webface
                working_path=WORKING_PATH,
                dataset_path=DATASET_PATH
            )

            