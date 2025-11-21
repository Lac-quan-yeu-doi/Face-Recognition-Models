import os
import sys
from contextlib import redirect_stdout
from utils.utils import Tee
from utils.criterion import *
from utils.config import DATASET_PATH, WORKING_PATH
from utils.model_utils import main_pipeline
import torchvision.transforms as transforms

model_dict = {
    1: {
        'log_file_name': 'sphereface.txt',
        'model_class': SphereFaceNet,
        'model_name': "SphereFace",
        'project_name': "sphereface-training",
        'model_final_filename': "sphereface_final.pth",
        'model_best_filename': "sphereface_best.pth"
    },
    2: {
        'log_file_name': 'cosface.txt',
        'model_class': CosFaceNet,
        'model_name': "CosFace",
        'project_name': "cosface-training",
        'model_final_filename': "cosface_final.pth",
        'model_best_filename': "cosface_best.pth"
    },
    3: {
        'log_file_name': 'arcface.txt',
        'model_class': ArcFaceNet,
        'model_name': "ArcFace",
        'project_name': "arcface-training",
        'model_final_filename': "arcface_final.pth",
        'model_best_filename': "arcface_best.pth"
    },
    4: {
        'log_file_name': 'mv_softmax_cos.txt',
        'model_class': MV_SoftmaxCosNet,
        'model_name': "MV_Softmax_cos",
        'project_name': "mv_softmax-training",
        'model_final_filename': "mv_softmax_cos_final.pth",
        'model_best_filename': "mv_softmax_cos_best.pth"
    },
    5: {
        'log_file_name': 'mv_softmax_arc.txt',
        'model_class': MV_SoftmaxArcNet,
        'model_name': "MV_Softmax_arc",
        'project_name': "mv_softmax-training",
        'model_final_filename': "mv_softmax_arc_final.pth",
        'model_best_filename': "mv_softmax_arc_best.pth"
    },
    6: {
        'log_file_name': 'curricularface.txt',
        'model_class': CurricularFaceNet,
        'model_name': "CurricularFace",
        'project_name': "curricularface-training",
        'model_final_filename': "curricularface_final.pth",
        'model_best_filename': "curricularface_best.pth"
    },
    7: {
        'log_file_name': 'vpl_arcface.txt',
        'model_class': VPLArcFaceNet,
        'model_name': "VPLArcFace",
        'project_name': "vpl_arcface-training",
        'model_final_filename': "vpl_arcface_final.pth",
        'model_best_filename': "vpl_arcface_best.pth"
    },
    8: {
        'log_file_name': 'magface.txt',
        'model_class': MagFaceNet,
        'model_name': "MagFace",
        'project_name': "magface-training",
        'model_final_filename': "magface_final.pth",
        'model_best_filename': "magface_best.pth"
    },
    9: {
        'log_file_name': 'adaface.txt',
        'model_class': AdaFaceNet,
        'model_name': "AdaFace",
        'project_name': "adaface-training",
        'model_final_filename': "adaface_final.pth",
        'model_best_filename': "adaface_best.pth"
    },
    10: {
        'log_file_name': 'elastic_cosface.txt',
        'model_class': ElasticCosFaceNet,
        'model_name': "ElasticCosFace",
        'project_name': "elasticface-training",
        'model_final_filename': "elastic_cosface_final.pth",
        'model_best_filename': "elastic_cosface_best.pth"
    },
    11: {
        'log_file_name': 'elastic_arcface.txt',
        'model_class': ElasticArcFaceNet,
        'model_name': "ElasticArcFace",
        'project_name': "elasticface-training",
        'model_final_filename': "elastic_arcface_final.pth",
        'model_best_filename': "elastic_arcface_best.pth"
    },
    12: {
        'log_file_name': 'sphereface2.txt',
        'model_class': SphereFace2Net,
        'model_name': "SphereFace2",
        'project_name': "sphereface2-training",
        'model_final_filename': "sphereface2_final.pth",
        'model_best_filename': "sphereface2_best.pth"
    },
    13: {
        'log_file_name': 'uniface.txt',
        'model_class': UniFaceNet,
        'model_name': "UniFace",
        'project_name': "uniface-training",
        'model_final_filename': "uniface_final.pth",
        'model_best_filename': "uniface_best.pth"
    },
    14: {
        'log_file_name': 'unitsface.txt',
        'model_class': UniTSFaceNet,
        'model_name': "UniTSFace",
        'project_name': "unitsface-training",
        'model_final_filename': "unitsface_final.pth",
        'model_best_filename': "unitsface_best.pth"
    },
    15: {
        'log_file_name': 'qaface.txt',
        'model_class': QAFaceNet,
        'model_name': "QAFace",
        'project_name': "qaface-training",
        'model_final_filename': "qaface_final.pth",
        'model_best_filename': "qaface_best.pth"
    },
    16: {
        'log_file_name': 'qmagface.txt',
        'model_class': QMagFaceNet,
        'model_name': "QMagFace",
        'project_name': "qmagface-training",
        'model_final_filename': "qmagface_final.pth",
        'model_best_filename': "qmagface_best.pth"
    }
}


model_numbers = {
    1: 'SphereFace',
    2: 'CosFace',
    3: 'ArcFace',
    4: 'MV_Softmax_cos',
    5: 'MV_Softmax_arc',
    6: 'CurricularFace',
    7: 'VPLArcFace',
    8: 'MagFace',
    9: 'AdaFace',
    10: 'ElasticCosFace',
    11: 'ElasticArcFace',
    12: 'SphereFace2',
    13: 'UniFace',
    14: 'UniTSFace',
    15: 'QAFace',
    16: 'QMagFace'
}

def run(
    log_file_name,
    model_class,
    model_name,
    project_name,
    model_final_filename,
    model_best_filename
):
    # Setup logging
    log_folder = f'{WORKING_PATH}/log'
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, log_file_name)
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main_pipeline(
                model_class=model_class,
                model_name=model_name,
                project_name=project_name,
                model_final_filename=model_final_filename,
                model_best_filename=model_best_filename,
                num_classes=10575,
                working_path=WORKING_PATH,
                dataset_path=DATASET_PATH
            )
            
if __name__ == '__main__':
    for num, name in model_numbers.items():
        print(f'({num}): {name}')
        
    mode = input("Choose model:")
    if int(mode) not in model_dict.keys():
        raise Exception("Invalid mode")
    config = model_dict[int(mode)]
    run(
        log_file_name=config['log_file_name'],
        model_class=config['model_class'],
        model_name=config['model_name'],
        project_name=config['project_name'],
        model_final_filename=config['model_final_filename'],
        model_best_filename=config['model_best_filename']
    )
    
    
    
    
    