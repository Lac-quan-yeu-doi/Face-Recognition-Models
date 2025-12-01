import os
import shutil
import pandas as pd
import torch
import torchvision.transforms as transforms
from utils.criterion import *
from utils.model_utils import cross_validate_kfold  
from utils.config import DATASET_PATH

test_names = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cplfw', 'xqlfw']
# test_names = ['lfw']

model_names = {
    'SphereFace': SphereFaceNet,
    'CosFace': CosFaceNet,
    'ArcFace': ArcFaceNet,
    'MV_Softmax_cos': MV_SoftmaxCosNet,
    'MV_Softmax_arc': MV_SoftmaxArcNet,
    'CurricularFace': CurricularFaceNet,
    'VPLArcFace': VPLArcFaceNet,
    'MagFace': MagFaceNet,
    'AdaFace': AdaFaceNet,
    'ElasticCosFace': ElasticCosFaceNet,
    # 'ElasticArcFace': ElasticArcFaceNet,
    'SphereFace2': SphereFace2Net,
    'UniFace': UniFaceNet,
    'UniTSFace': UniTSFaceNet,
    'QAFace': QAFaceNet,
    'QMagFace': QMagFaceNet
}

num_classes = 10575
backbone_name = "resnet18"
batch_size = 64

device       = 'cuda' if torch.cuda.is_available() else 'cpu'
model_folder = f"models_evaluation/{backbone_name}"
output_dir   = f"evaluation_results"
os.makedirs(output_dir, exist_ok=True)

acc_records = []
auc_records = []

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

for model_name, ModelClass in model_names.items():
    model_name = "MV_Softmax_am" if model_name == "MV_Softmax_cos" else model_name
    ckpt_path = f"{model_folder}/{model_name}_min_loss.pth"
    # ckpt_path = f"{model_folder}/{model_name.lower()}_final.pth"
    if not os.path.exists(ckpt_path):
        print(f"[Warning] {ckpt_path} not found → skipping model")
        continue
    try:
        model = ModelClass(num_classes=num_classes, backbone=backbone_name)
    except Exception as e:
        print(f"[Error] Failed to instantiate {model_name}: {e}")
        continue

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint {ckpt_path}")

    acc_row = {"model": model_name}
    auc_row = {"model": model_name}

    for test in test_names:
        pairs_file = f'{DATASET_PATH}/face_evaluation_data/{test}/pair.list'
        img_dir    = f'{DATASET_PATH}/face_evaluation_data/{test}/imgs'

        if not os.path.exists(pairs_file) or not os.path.isdir(img_dir):
            print(f"[Warning] Data for {test} not found → skipping")
            acc_row[test] = "-"
            auc_row[test] = "-"
            continue

        print(f"\n=== {model_name} → {test} ===")
        mean_acc, std_acc, mean_auc, std_auc = cross_validate_kfold(
            model=model,
            pairs_file=pairs_file,
            img_dir=img_dir,
            transform=test_transform,
            device=device,
            batch_size=batch_size,
            k_fold=10
        )

        # Store *mean* values (you can also keep the std if you want)
        acc_row[test] = f"{mean_acc:.4f}"
        auc_row[test] = f"{mean_auc:.4f}"

        print(f"  Acc : {mean_acc:.4f}% ± {std_acc:.4f}%")
        print(f"  AUC : {mean_auc:.4f} ± {std_auc:.4f}")

    acc_records.append(acc_row)
    auc_records.append(auc_row)

columns = ["model"] + test_names

df_acc = pd.DataFrame(acc_records, columns=columns)
df_auc = pd.DataFrame(auc_records, columns=columns)

df_acc.set_index("model", inplace=True)
df_auc.set_index("model", inplace=True)

df_acc.to_csv(os.path.join(output_dir, f"accuracy_10fold_{backbone_name}.csv"))
df_auc.to_csv(os.path.join(output_dir, f"auc_10fold_{backbone_name}.csv"))

print("\n" + "="*60)
print("ACCURACY".center(60))
print("="*60)
print(df_acc)
print("\n" + "="*60)
print("AUC".center(60))
print("="*60)
print(df_auc)
print("\nResults saved to:", output_dir)

