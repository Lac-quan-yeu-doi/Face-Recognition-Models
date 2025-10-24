import os
import sys
import time
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torch.amp import GradScaler
from contextlib import redirect_stdout

from utils.criterion import CosFaceNet
from utils.dataset import CASIAwebfaceDataset, LFWPairDataset
from utils.model_utils import train_model, tune_threshold, evaluate_lfw_10fold, custom_collate_fn, parse_args
from utils.utils import Tee
from utils.config import *


def main():
    start_time = time.time()
    print(f"{'*'*10} COSFACE MODEL {'*'*10}")

    # Path
    model_folder_path = f'{WORKING_PATH}/models'
    os.makedirs(model_folder_path, exist_ok=True)
    model_final_path = f'{model_folder_path}/cosface_final.pth'
    model_best_path = f'{model_folder_path}/cosface_best.pth'

    # Hyperparameters
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using: ", device)

    # Data Transforms
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    train_dataset = CASIAwebfaceDataset(root_dir=f'{DATASET_PATH}/CASIA-webface', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)
    print(f"{'#'*20}  Data loaded {'#'*20}")

    # Initialize Model
    model = CosFaceNet(num_classes=train_dataset.num_of_identities).to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    scaler = GradScaler()

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        train_loss = train_model(model, train_loader, optimizer, scaler, device, epoch, num_epochs)
        print(f'[Epoch {epoch}/{num_epochs}] Train Loss: {train_loss:.6f} - Current LR: {optimizer.param_groups[0]["lr"]:.4f}')
        scheduler.step()

    # Combine pairsDevTrain and pairsDevTest for threshold tuning
    train_pairs_dataset = LFWPairDataset(root_dir=f'{DATASET_PATH}/Labeled Faces in the Wild (LFW)', 
                                        pairs_file=f'{DATASET_PATH}/Labeled Faces in the Wild (LFW)/pairsDevTrain.txt', 
                                        transform=transform)
    test_pairs_dataset = LFWPairDataset(root_dir=f'{DATASET_PATH}/Labeled Faces in the Wild (LFW)', 
                                       pairs_file=f'{DATASET_PATH}/Labeled Faces in the Wild (LFW)/pairsDevTest.txt', 
                                       transform=transform)
    combined_pairs_dataset = ConcatDataset([train_pairs_dataset, test_pairs_dataset])
    
    # Tune threshold
    best_threshold, best_accuracy = tune_threshold(model, combined_pairs_dataset, batch_size, device)
    torch.save(model.state_dict(), model_best_path)
    print(f'Best Threshold: {best_threshold:.2f} - Combined DevTrain+DevTest Accuracy: {best_accuracy:.2f}%')

    # 10-Fold Cross-Validation
    mean_accuracy, std_accuracy = evaluate_lfw_10fold(model, 
                                                     pairs_file=f'{DATASET_PATH}/Labeled Faces in the Wild (LFW)/pairs.txt',
                                                     batch_size=batch_size,
                                                     root_dir=f'{DATASET_PATH}/Labeled Faces in the Wild (LFW)',
                                                     transform=transform,
                                                     device=device,
                                                     threshold=best_threshold)
    print(f'LFW 10-Fold Verification Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}% (Threshold: {best_threshold:.2f})')

    # Save Final Model
    torch.save(model.state_dict(), model_final_path)

    end_time = time.time()
    print(f"Code runs in {end_time - start_time:.2f}s")

if __name__ == '__main__':
    log_folder = f'{WORKING_PATH}/log'
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, 'cosface.txt')
    with open(log_file_path, 'w') as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            main()