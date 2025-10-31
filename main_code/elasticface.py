# ver 2

import os
import sys
from contextlib import redirect_stdout
from utils.utils import Tee
from utils.criterion import ElasticArcFaceNet, ElasticCosFaceNet
from utils.config import DATASET_PATH, WORKING_PATH
from utils.model_utils import main_pipeline
import torchvision.transforms as transforms

if __name__ == '__main__':
    # Setup logging
    log_folder = f'{WORKING_PATH}/log'
    os.makedirs(log_folder, exist_ok=True)

    base_type = 'arc' # 'arc' / 'cos

    if base_type == 'arc':
        log_file_path = os.path.join(log_folder, 'elastic_arcface.txt')
        with open(log_file_path, 'w') as log_file:
            with redirect_stdout(Tee(sys.stdout, log_file)):
                main_pipeline(
                    model_class=ElasticArcFaceNet,
                    model_name="ElasticArcFace",
                    project_name="elasticface-training",
                    model_final_filename="elastic_arcface_final.pth",
                    model_best_filename="elastic_arcface_best.pth",
                    num_classes=10575, # 10575 for CASIA-webface
                    working_path=WORKING_PATH,
                    dataset_path=DATASET_PATH
                )
    else:
        log_file_path = os.path.join(log_folder, 'elastic_cosface.txt')
        with open(log_file_path, 'w') as log_file:
            with redirect_stdout(Tee(sys.stdout, log_file)):
                main_pipeline(
                    model_class=ElasticCosFaceNet,
                    model_name="ElasticCosFace",
                    project_name="elasticface-training",
                    model_final_filename="elastic_cosface_final.pth",
                    model_best_filename="elastic_cosface_best.pth",
                    num_classes=10575, # 10575 for CASIA-webface
                    working_path=WORKING_PATH,
                    dataset_path=DATASET_PATH
                )