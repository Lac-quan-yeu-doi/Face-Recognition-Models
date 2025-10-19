# ============================================================
# Standard Library Imports
# ============================================================
import os
import sys
import time
import shutil
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

from utils.config import *
from utils.dataset import CASIAwebfaceDataset, LFWPairDataset
from utils.optimizers import get_optimizer
from utils.schedulers import *

load_dotenv()

def save_checkpoint(model, optimizer, scheduler, scaler, train_loss, epoch, model_checkpoints_path, model_name, isCheckpoint=True):
    """
    Save a checkpoint or minimum-loss model state, keeping up to 3 latest epoch-based checkpoints.

    Args:
        model: The PyTorch model to save.
        optimizer: The optimizer state to save.
        scheduler: The scheduler state to save.
        scaler: The GradScaler state for mixed precision training.
        epoch: Current training epoch.
        model_checkpoints_path (str): Directory to save checkpoints.
        model_name (str): Base name for checkpoint files.
        isCheckpoint (bool): If True, save as epoch-based checkpoint; else, save as min-loss model.
    """
    os.makedirs(model_checkpoints_path, exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'train_loss': train_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
    }
    
    if isCheckpoint:
        checkpoint_path = f'{model_checkpoints_path}/{model_name}_checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only the 3 latest epoch-based checkpoints
        checkpoints = sorted(
            [f for f in os.listdir(model_checkpoints_path) 
             if f.startswith(f'{model_name}_checkpoint_epoch_') and f.endswith('.pth')],
            key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0])
        )
        while len(checkpoints) > 3:
            os.remove(os.path.join(model_checkpoints_path, checkpoints.pop(0)))
    else:
        checkpoint_path = f'{model_checkpoints_path}/{model_name}_min_loss.pth'
        torch.save(checkpoint, checkpoint_path)

def load_latest_checkpoint(model, optimizer, scheduler, scaler, model_checkpoints_path, model_name, device, isCheckpoint=True):
    """
    Load the latest epoch-based checkpoint or the minimum-loss model if available.

    Args:
        model: The PyTorch model to load state into.
        optimizer: The optimizer to load state into.
        scheduler: The scheduler to load state into.
        scaler: The GradScaler to load state into.
        model_checkpoints_path (str): Directory containing checkpoints.
        model_name (str): Base name for checkpoint files.
        device: Device to map the checkpoint to (e.g., 'cuda' or 'cpu').
        isCheckpoint (bool): If True, load latest epoch-based checkpoint; else, load min-loss model.

    Returns:
        int: The epoch to start training from (epoch + 1 from checkpoint, or 1 if none found).
    """
    if not os.path.exists(model_checkpoints_path):
        return 1, None

    if isCheckpoint:
        checkpoints = sorted(
            [f for f in os.listdir(model_checkpoints_path) 
             if f.startswith(f'{model_name}_checkpoint_epoch_') and f.endswith('.pth')],
            key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]),
            reverse=True
        )
        checkpoint_name = 'last checkpoint'
    else:
        # remove the old checkpoints because min_loss may not be the latest
        for f in os.listdir(model_checkpoints_path):
            if f.startswith(f"{model_name}_checkpoint_epoch_") and f.endswith(".pth"):
                print(f"Remove {f}")
                os.remove(os.path.join(model_checkpoints_path, f))
        print("Remove old checkpoints")

        checkpoints = [f for f in os.listdir(model_checkpoints_path) 
                       if f == f'{model_name}_min_loss.pth']
        checkpoint_name = 'min_loss_model'

    if checkpoints:
        latest_checkpoint = os.path.join(model_checkpoints_path, checkpoints[0])
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        train_loss = checkpoint.get('train_loss', None)
        print(f"### Resuming training from {checkpoint_name} - epoch {checkpoint['epoch']} - {latest_checkpoint} ###")
        return start_epoch, train_loss

    return 1, None  # Start from epoch 1 if no checkpoint is found

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

def train_model(model, train_loader, optimizer, scaler, device, epoch, epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
    for batch in loop:
        if batch is None:
            continue
        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(device_type='cuda'):
            loss, _ = model(images, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    wandb.log({"train_loss": avg_loss, "epoch": epoch}) 
    return avg_loss

def evaluate_model(model, test_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(test_loader, desc="Evaluating (Classification)", leave=False)
        for batch in loop:
            if batch is None:
                continue
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            features = model(images)
            # Compute ArcFace loss
            loss = model.arcface(features, labels)
            total_loss += loss.item()
            # Compute predictions using ArcFace scores
            features = nn.functional.normalize(features, p=2, dim=1)
            w = nn.functional.normalize(model.arcface.weight, p=2, dim=1)
            cos_theta = torch.mm(features, w.t())
            theta = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            one_hot = nn.functional.one_hot(labels, num_classes=model.arcface.num_classes)
            cos_theta_m = torch.cos(theta + model.arcface.m * one_hot)
            scaled_logits = model.arcface.s * torch.where(one_hot.bool(), cos_theta_m, cos_theta)
            _, predicted = torch.max(scaled_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    wandb.log({"val_loss": avg_loss, "val_accuracy": accuracy}) 
    return avg_loss, accuracy

def evaluate_lfw_verification(model, pairs_dataset, batch_size, device, threshold=0.5):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(DataLoader(pairs_dataset, batch_size=batch_size, shuffle=False, num_workers=8, collate_fn=custom_collate_fn),
                    desc="Evaluating (Verification)", leave=False)
        for batch in loop:
            if batch is None:
                continue
            img1, img2, same = batch
            img1, img2, same = img1.to(device), img2.to(device), same.to(device)
            feat1 = model(img1)
            feat2 = model(img2)
            cos_sim = nn.functional.cosine_similarity(feat1, feat2)
            pred = (cos_sim > threshold).long()
            correct += (pred == same).sum().item()
            total += same.size(0)
    accuracy = 100 * correct / total
    return accuracy

def evaluate_lfw_10fold(model, pairs_file, batch_size, root_dir, transform, device, threshold=0.5):
    model.eval()
    fold_accuracies = []
    
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split()
        num_folds, num_same, num_diff = map(int, header)
        pairs_per_fold = num_same + num_diff
        lines = lines[1:]
    
    for fold in range(num_folds):
        start_idx = fold * pairs_per_fold
        end_idx = (fold + 1) * pairs_per_fold
        fold_pairs = []
        same_count, diff_count = 0, 0
        for line in lines[start_idx:end_idx]:
            parts = line.strip().split()
            if len(parts) == 3:
                name, img1, img2 = parts
                fold_pairs.append((f"{name}/{name}_{img1.zfill(4)}.jpg", f"{name}/{name}_{img2.zfill(4)}.jpg", 1))
                same_count += 1
            elif len(parts) == 4:
                name1, img1, name2, img2 = parts
                fold_pairs.append((f"{name1}/{name1}_{img1.zfill(4)}.jpg", f"{name2}/{name2}_{img2.zfill(4)}.jpg", 0))
                diff_count += 1
        
        temp_dataset = LFWPairDataset(root_dir, pairs_file, transform)
        temp_dataset.pairs = fold_pairs
        accuracy = evaluate_lfw_verification(model, temp_dataset, batch_size, device, threshold)
        fold_accuracies.append(accuracy)
        print(f'Fold {fold + 1} Verification Accuracy: {accuracy:.2f}%')

    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    return mean_accuracy, std_accuracy

def tune_threshold(model, pairs_dataset, batch_size, device, thresholds=np.arange(0.0, 0.35, 0.01)):
    best_threshold = 0.5
    best_accuracy = 0
    threshold_list = []
    accuracy_list = []
    print("Tuning verification threshold...")
    for thresh in thresholds:
        accuracy = evaluate_lfw_verification(model, pairs_dataset, batch_size, device, thresh)
        threshold_list.append(float(thresh))
        accuracy_list.append(float(accuracy))
        print(f'Threshold {thresh:.2f}: Verification Accuracy {accuracy:.2f}%')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    return best_threshold, best_accuracy, threshold_list, accuracy_list

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument(
        '--continue_train',
        choices=['min_loss', 'latest'],
        help=(
            "Resume training:\n"
            "  min_loss  -> resume from best checkpoint\n"
            "  latest    -> resume from latest checkpoint\n"
            "If not provided, training starts from scratch."
        ),
    )
    parser.add_argument(
        '--model-save-path',
        type=str,
        default=f'{WORKING_PATH}/models',
        help='Path to save model checkpoints'
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='face-recognition-training',
        help='W&B project name'
    )

    return parser.parse_args()

def main_pipeline(
    model_class,
    model_name,
    project_name,
    model_final_filename,
    model_best_filename,
    num_classes,
    working_path,
    dataset_path,
):
    """
    General function to train and evaluate a face recognition model.
    [[]]
    Args:
        model_class: The class of the model (e.g., ArcFaceNet, CosFaceNet).
        model_name: Name of the model for logging (e.g., "ArcFace").
        project_name: WandB project name (e.g., "arcface-training").
        model_final_filename: Filename for the final model (e.g., "arcface_final.pth").
        model_best_filename: Filename for the best model (e.g., "arcface_best.pth").
        num_classes: Number of classes (identities) in the dataset.
        working_path: Base path for saving models and logs.
        dataset_path: Base path for datasets.
    """
    # Initialize WandB
    env_path = Path("../.env")
    load_dotenv(dotenv_path=env_path)
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
    wandb.login(key=WANDB_API_KEY)

    start_time = time.time()
    print(f"{'*'*10} {model_name.upper()} MODEL {'*'*10}")

    # Hyperparameters
    args = parse_args()
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr
    continue_train = args.continue_train
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"### Training with batch size {batch_size} - epochs {num_epochs} - lr {learning_rate} ###")
    print(f"Using: {device}")

    # Training setting setup
    isCheckpoint = False
    if continue_train is None:
        print("🧠 Training from scratch...")
    elif continue_train == 'min_loss':
        print("🔁 Resuming from best min_loss checkpoint...")
    elif continue_train == 'latest':
        isCheckpoint = True
        print("🔁 Resuming from latest checkpoint...")
    else:
        raise Exception(f"Unknown value {continue_train}")

    # Path setup
    model_checkpoints_path = CHECKPOINTS_FOLDER_PATH
    if (continue_train is None) and os.path.exists(model_checkpoints_path):
        shutil.rmtree(model_checkpoints_path)
        print("Training from scratch, reset all checkpoints...")
    os.makedirs(model_checkpoints_path, exist_ok=True)
    model_final_path = f'{model_checkpoints_path}/{model_final_filename}'
    model_best_path = f'{model_checkpoints_path}/{model_best_filename}'
    log_folder = f'{working_path}/log'
    os.makedirs(log_folder, exist_ok=True)
    log_file_path = os.path.join(log_folder, f'{model_name.lower()}.txt')
    
    # --- Initialize WandB ---
    wandb_config = {
        "batch_size": batch_size,
        "epochs": num_epochs,
        "learning_rate": learning_rate,
        "optimizer": "SGD",
        "scheduler": "CustomStep",
        "model": model_name,
    }

    run = wandb.init(
        project=project_name,
        config=wandb_config,
        dir=WORKING_PATH
        )

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load Datasets
    train_dataset = CASIAwebfaceDataset(
        root_dir=f'{dataset_path}/CASIA-webface',
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, collate_fn=custom_collate_fn
    )
    print(f"{'#'*20} Data loaded {'#'*20}")

    # Initialize Model
    model = model_class(num_classes=num_classes).to(device)

    optimizer = get_optimizer(model, "sgd")
    scheduler = get_scheduler(optimizer, "customstep")

    scaler = GradScaler()

    # Load latest checkpoint if available
    start_epoch, min_train_loss = load_latest_checkpoint(model, optimizer, scheduler, scaler, model_checkpoints_path, model_name, device, isCheckpoint)
    optimizer.param_groups[0]['lr'] = learning_rate
    if isinstance(scheduler, CustomStepLR):
        scheduler.last_epoch = start_epoch

    # Watch model in W&B
    wandb.watch(model, log="all", log_freq=100)

    # Training Loop
    if min_train_loss is None:
        min_train_loss = np.inf
    print(f"### Train with min_train_loss = {min_train_loss} ### ")
    for epoch in range(start_epoch, num_epochs + start_epoch):
        train_loss = train_model(model, train_loader, optimizer, scaler, device, epoch, num_epochs + start_epoch - 1)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f'[Epoch {epoch}/{num_epochs + start_epoch - 1}] Train Loss: {train_loss:.6f} - Current LR: {current_lr:.6f}')
        wandb.log({"epoch": epoch, "train_loss": train_loss, "learning_rate": current_lr}, step=epoch)

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, scaler, train_loss, epoch, model_checkpoints_path, model_name, isCheckpoint)
            print("🤖 Save model on min loss")  

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, scheduler, scaler, train_loss, epoch, model_checkpoints_path, model_name, True)
        
        scheduler.step()

    # Combine pairsDevTrain and pairsDevTest for threshold tuning
    train_pairs_dataset = LFWPairDataset(
        root_dir=f'{dataset_path}/Labeled Faces in the Wild (LFW)',
        pairs_file=f'{dataset_path}/Labeled Faces in the Wild (LFW)/pairsDevTrain.txt',
        transform=transform
    )
    test_pairs_dataset = LFWPairDataset(
        root_dir=f'{dataset_path}/Labeled Faces in the Wild (LFW)',
        pairs_file=f'{dataset_path}/Labeled Faces in the Wild (LFW)/pairsDevTest.txt',
        transform=transform
    )
    combined_pairs_dataset = ConcatDataset([train_pairs_dataset, test_pairs_dataset])

    # Tune threshold
    best_threshold, best_accuracy, threshold_list, accuracy_list = tune_threshold(model, combined_pairs_dataset, batch_size, device)
    torch.save(model.state_dict(), model_best_path)
    print(f'Best Threshold: {best_threshold:.2f} - Combined DevTrain+DevTest Accuracy: {best_accuracy:.2f}%')
    threshold_acc_table = wandb.Table(
        data=[[t, a] for t, a in zip(threshold_list, accuracy_list)],
        columns=["threshold", "accuracy"]
    )
    wandb.log({
        "Threshold vs Accuracy": wandb.plot.line(
            threshold_acc_table,
            "threshold",
            "accuracy",
            title="Threshold vs Accuracy Curve"
        )
    })

    # 10-Fold Cross-Validation
    mean_accuracy, std_accuracy = evaluate_lfw_10fold(
        model,
        pairs_file=f'{dataset_path}/Labeled Faces in the Wild (LFW)/pairs.txt',
        batch_size=batch_size,
        root_dir=f'{dataset_path}/Labeled Faces in the Wild (LFW)',
        transform=transform,
        device=device,
        threshold=best_threshold
    )
    print(f'LFW 10-Fold Verification Accuracy: {mean_accuracy:.2f}% ± {std_accuracy:.2f}% (Threshold: {best_threshold:.2f})')

    # Save Final Model
    torch.save(model.state_dict(), model_final_path)
    wandb.save(model_final_path)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Code runs in {total_time:.2f}s")
    
    artifact = wandb.Artifact("checkpoints", type="model")
    artifact.add_dir(model_checkpoints_path)
    run.log_artifact(artifact)
    print("✅ CHECKPOINTS UPLOADED")

    run.finish()

    return model, best_threshold, best_accuracy, mean_accuracy, std_accuracy


