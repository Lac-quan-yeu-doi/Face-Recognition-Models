import os
import sys
from contextlib import redirect_stdout
from utils.utils import Tee
from utils.criterion import MV_SoftmaxNet
import torchvision.transforms as transforms
from utils.config import DATASET_PATH, WORKING_PATH, MARGIN_TYPE_mv
from utils.model_utils import main_pipeline

if __name__ == '__main__':
    # Setup logging
    log_folder = f'{WORKING_PATH}/log'
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, 'mv_softmax_{MARGIN_TYPE_mv}.txt')
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main_pipeline(
                model_class=MV_SoftmaxNet,
                model_name=f"MV_Softmax_{MARGIN_TYPE_mv}",
                project_name=f"mv_softmax-training",
                model_final_filename=f"mv_softmax_{MARGIN_TYPE_mv}_final.pth",
                model_best_filename=f"mv_softmax_{MARGIN_TYPE_mv}_best.pth",
                num_classes=10575,
                working_path=WORKING_PATH,
                dataset_path=DATASET_PATH
            )