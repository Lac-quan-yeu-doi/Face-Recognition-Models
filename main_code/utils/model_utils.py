# ============================================================
# Standard Library Imports
# ============================================================
import os
import sys
import csv
import time
import math
import shutil
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
try:
    # For PyTorch 2.0 and above
    from torch.amp import GradScaler, autocast
    autocast_context = autocast(device_type='cuda')
except ImportError:
    # For PyTorch 1.6 to 1.13
    from torch.cuda.amp import GradScaler, autocast
    autocast_context = autocast()
# from torch.amp import GradScaler, autocast
from tqdm import tqdm
import wandb
from dotenv import load_dotenv

from utils.config import *
from utils.dataset import CASIAwebfaceDataset, LFWPairDataset, FlatPairDataset
from utils.optimizers import get_optimizer
from utils.schedulers import *
from utils.metrics import accuracy
from utils.utils import AverageMeter, ProgressMeter

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
        optimizer: The optimizer to load state into. Update if not None
        scheduler: The scheduler to load state into. Update if not None
        scaler: The GradScaler to load state into. Update if not None
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
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None: 
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None:
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


def train_model(model, train_loader, criterion, optimizer, scaler, device, epoch, epochs, args):
    model.train()
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    losses_id = AverageMeter('L_ID', ':.3f')
    losses_mag = AverageMeter('L_mag', ':.6f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    throughputs = AverageMeter('ThroughPut', ':.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, throughputs, 'images/s', losses, losses_id, losses_mag, top1, top5],
        prefix=f"Epoch: [{epoch}/{epochs}]"
    )

    end = time.time()
    global iters
    iters = iters + 1 if 'iters' in globals() else 0

    for i, (images, target) in enumerate(train_loader):
        if images is None:
            continue

        data_time.update(time.time() - end)
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(device_type='cuda'):
            output, norm, loss_g, one_hot = model(images, target)
            cosine_s, logits = output
            loss_id = criterion(logits, target)
            loss = loss_id + args.lambda_g * loss_g

        acc1, acc5 = accuracy(cosine_s, target, topk=(1, 5))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        losses_id.update(loss_id.item(), batch_size)
        mag_loss = args.lambda_g * loss_g.item() if isinstance(loss_g, torch.Tensor) else args.lambda_g * loss_g
        losses_mag.update(mag_loss, batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)
        batch_time.update(time.time() - end)
        end = time.time()
        throughputs.update(batch_size / (time.time() - end + 1e-8))

        if i % args.print_freq == 0:
            progress.display(i)

        wandb.log({
            "loss": loss.item(),
            "loss_id": loss_id.item(),
            "loss_mag": mag_loss,
            "acc1": acc1[0],
            "acc5": acc5[0],
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch,
            "step": iters
        }, step=iters)

        iters += 1

    return losses.avg

# def evaluate_lfw_verification(model, pairs_dataset, batch_size, device, threshold=0.33):
#     model.eval()
#     correct = total = 0
#     with torch.no_grad():
#         loader = DataLoader(
#             pairs_dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=8,
#             collate_fn=custom_collate_fn,
#             pin_memory=True
#         )
#         for img1, img2, same in loader:
#             if img1 is None:
#                 continue
#             img1, img2, same = img1.to(device), img2.to(device), same.to(device)

#             # === CRITICAL: L2-normalize ===
#             feat1 = F.normalize(model(img1), dim=1)
#             feat2 = F.normalize(model(img2), dim=1)
#             cos_sim = (feat1 * feat2).sum(dim=1)

#             pred = (cos_sim > threshold).long()
#             correct += (pred == same).sum().item()
#             total += same.size(0)

#     return 100.0 * correct / total if total > 0 else 0.0

# def cross_validate_kfold(model, pairs_file, batch_size, root_dir, transform, device, threshold=0.33):
#     model.eval()
#     fold_accuracies = []
#     num_folds = 10

#     # Load pairs
#     all_pairs = []
#     with open(pairs_file, 'r') as f:
#         reader = csv.reader(f)
#         next(reader, None)
#         for row in reader:
#             if len(row) == 4 and (row[-1] in ['', ' ']):
#                 row = row[:-1]
#             if len(row) == 3:
#                 name, img1, img2 = row
#                 p1 = f"{name}/{name}_{img1.zfill(4)}.jpg"
#                 p2 = f"{name}/{name}_{img2.zfill(4)}.jpg"
#                 all_pairs.append((p1, p2, 1))
#             elif len(row) == 4:
#                 n1, i1, n2, i2 = row
#                 p1 = f"{n1}/{n1}_{i1.zfill(4)}.jpg"
#                 p2 = f"{n2}/{n2}_{i2.zfill(4)}.jpg"
#                 all_pairs.append((p1, p2, 0))

#     matched = [p for p in all_pairs if p[2] == 1]
#     mismatched = [p for p in all_pairs if p[2] == 0]

#     n_per_fold = min(len(matched), len(mismatched)) // num_folds
#     if n_per_fold == 0:
#         raise ValueError("Not enough pairs")

#     np.random.seed(42)
#     np.random.shuffle(matched)
#     np.random.shuffle(mismatched)

#     for fold in range(num_folds):
#         start = fold * n_per_fold
#         end = (fold + 1) * n_per_fold
#         fold_pairs = matched[start:end] + mismatched[start:end]
#         if not fold_pairs:
#             continue

#         dataset = LFWPairDataset(root_dir=root_dir, pairs_files=None, transform=transform)
#         dataset.pairs = fold_pairs

#         acc = evaluate_lfw_verification(model, dataset, batch_size, device, threshold)
#         fold_accuracies.append(acc)
#         print(f"  Fold {fold+1}: {acc:.3f}%")

#     mean_acc = np.mean(fold_accuracies)
#     std_acc = np.std(fold_accuracies)
#     print(f"10-fold: {mean_acc:.3f}% ± {std_acc:.3f}%")
#     return mean_acc, std_acc

# def tune_threshold(model, pairs_dataset, batch_size, device, thresholds=np.arange(0.1, 0.6, 0.005)):
#     """
#     Tune threshold on a SINGLE validation set (e.g., DevTrain + DevTest).
#     """
#     print("Tuning threshold on validation set...")
#     best_thresh = 0.33
#     best_acc = 0.0
#     results = []

#     for thresh in thresholds:
#         acc = evaluate_lfw_verification(model, pairs_dataset, batch_size, device, thresh)
#         results.append((thresh, acc))
#         print(f"  Threshold {thresh:.3f} → {acc:.3f}%")
#         if acc > best_acc:
#             best_acc = acc
#             best_thresh = thresh

#     print(f"Best threshold: {best_thresh:.4f} → {best_acc:.3f}%")
#     return best_thresh, best_acc, results

def compute_auc(model, dataset, batch_size, device):
    """
    Compute AUC on the given dataset.
    """
    model.eval()
    all_similarities = []
    all_labels = []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for img1, img2, same in loader:
            if img1 is None:
                continue
            img1, img2 = img1.to(device), img2.to(device)
            
            feat1 = F.normalize(model(img1), dim=1)
            feat2 = F.normalize(model(img2), dim=1)
            cos_sim = (feat1 * feat2).sum(dim=1)
            
            all_similarities.extend(cos_sim.cpu().numpy())
            if isinstance(same, torch.Tensor):
                all_labels.extend(same.cpu().numpy())
            else:
                all_labels.extend(same)
    
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    if len(np.unique(all_labels)) < 2:
        return 0.0  # AUC undefined if only one class
    
    return roc_auc_score(all_labels, all_similarities)
    
def evaluate(model, dataset, batch_size, device, threshold=0.33):
    model.eval()
    correct = total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for img1, img2, same in loader:
            if img1 is None:
                continue
            img1, img2 = img1.to(device), img2.to(device)
            # Convert same to tensor if needed
            if not isinstance(same, torch.Tensor):
                same = torch.tensor(same, device=device)
            else:
                same = same.to(device)
            
            feat1 = F.normalize(model(img1), dim=1)
            feat2 = F.normalize(model(img2), dim=1)
            cos_sim = (feat1 * feat2).sum(dim=1)
            pred = (cos_sim > threshold).long()
            correct += (pred == same).sum().item()
            total += same.size(0)

    return 100.0 * correct / total if total > 0 else 0.0

def tune_threshold_roc(model, dataset, batch_size, device):    
    model.eval()
    all_similarities = []
    all_labels = []
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    with torch.no_grad():
        for img1, img2, same in loader:
            if img1 is None:
                continue
            img1, img2 = img1.to(device), img2.to(device)
            feat1 = F.normalize(model(img1), dim=1)
            feat2 = F.normalize(model(img2), dim=1)
            cos_sim = (feat1 * feat2).sum(dim=1)
            
            all_similarities.extend(cos_sim.cpu().numpy())
            # Handle both tensor and list inputs
            if isinstance(same, torch.Tensor):
                all_labels.extend(same.cpu().numpy())
            else:
                all_labels.extend(same)
    
    all_similarities = np.array(all_similarities)
    all_labels = np.array(all_labels)
    
    # Find optimal threshold using Youden's index (maximizes TPR - FPR)
    fpr, tpr, thresholds = roc_curve(all_labels, all_similarities)
    optimal_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[optimal_idx]
    
    # Calculate accuracy at this threshold
    predictions = (all_similarities > best_thresh).astype(int)
    best_acc = 100.0 * (predictions == all_labels).sum() / len(all_labels)
    
    return best_thresh, best_acc

def cross_validate_kfold(model, pairs_file, img_dir, transform, device, batch_size=64, k_fold=10):
    """
    10-fold evaluation on LFW-style .list file.
    Returns: mean_acc, std_acc, mean_auc, std_auc
    """
    # ---- Load pairs ----
    all_pairs, labels = [], []
    with open(pairs_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            a, b, label = int(parts[0]), int(parts[1]), int(parts[2])
            all_pairs.append((a, b, label))
            labels.append(label)

    all_pairs = np.array(all_pairs)
    labels = np.array(labels)

    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_pairs, labels), 1):
        val_pairs = all_pairs[val_idx]
        train_pairs = all_pairs[train_idx]

        val_dataset = FlatPairDataset(val_pairs, img_dir, transform)
        train_dataset = FlatPairDataset(train_pairs, img_dir, transform)

        print(f"\n=== Fold {fold}/{k_fold} ===")
        
        # Tune threshold on validation fold
        best_thresh, _ = tune_threshold_roc(model, val_dataset, batch_size, device)
        print(f"Best threshold: {best_thresh:.4f}")

        # Evaluate accuracy on train (test proxy) using tuned threshold
        acc = evaluate(model, train_dataset, batch_size, device, best_thresh)
        fold_accuracies.append(acc)
        print(f"Accuracy (on k-1 folds): {acc:.3f}%")

        # Compute AUC on test set (train folds)
        auc = compute_auc(model, train_dataset, batch_size, device)
        fold_aucs.append(auc)
        print(f"AUC (on k-1 folds): {auc:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)

    print(f"\n{k_fold}-fold Results:")
    print(f"Accuracy: {mean_acc:.3f}% ± {std_acc:.3f}%")
    print(f"AUC:      {mean_auc:.4f} ± {std_auc:.4f}")

    return mean_acc, std_acc, mean_auc, std_auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=512)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    # parser.add_argument('--backbone', '-bb', type=str, default='resnet18')
    parser.add_argument('--lambda_g', type=float, default=0.0, help="Magnitude loss weight")
    parser.add_argument('--print_freq', type=int, default=100)
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
    dataset_path
):
    start_time = time.time()
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === W&B ===
    wandb.init(
        project=project_name,
        name=model_name,
        config=vars(args),
        dir=f'{WORKING_PATH}/wandb'
    )

    # === Paths ===
    model_checkpoints_path = f"{working_path}/checkpoints/{model_name}"
    if (args.continue_train is None) and os.path.exists(model_checkpoints_path):
        shutil.rmtree(model_checkpoints_path)
        print("Training from scratch, reset all checkpoints...")
    os.makedirs(model_checkpoints_path, exist_ok=True)

    print(f"Training using {device} - batch size {args.batch_size} - epochs {args.epochs} - learning rate {args.learning_rate}")
    # === Data ===
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_dataset_1 = CASIAwebfaceDataset(root_dir=f"{dataset_path}/CASIA-WebFace", split='train', transform=train_transform)
    train_dataset_2 = CASIAwebfaceDataset(root_dir=f"{dataset_path}/CASIA-WebFace", split='valid', transform=train_transform)
    train_dataset = ConcatDataset([train_dataset_1, train_dataset_2])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)

    # === Model, Opt, Scheduler ===
    model = model_class(num_classes=num_classes, backbone=BACKBONE).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = get_scheduler(optimizer, "customstep")
    scaler = GradScaler()

    # === Resume ===
    start_epoch, min_train_loss = load_latest_checkpoint(
        model, optimizer, scheduler, scaler, model_checkpoints_path, model_name, device, isCheckpoint=(args.continue_train == 'latest')
    )
    if min_train_loss is None:
        min_train_loss = float('inf')

    # === Training Loop ===
    for epoch in range(start_epoch, args.epochs + start_epoch):
        train_loss = train_model(model, train_loader, criterion, optimizer, scaler, device, epoch, args.epochs + start_epoch - 1, args)

        if train_loss < min_train_loss:
            min_train_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, scaler, train_loss, epoch, model_checkpoints_path, model_name, isCheckpoint=False)
            print(f"New best model saved: {train_loss:.6f}")

        save_checkpoint(model, optimizer, scheduler, scaler, train_loss, epoch, model_checkpoints_path, model_name, isCheckpoint=True)
        scheduler.step()

    # Save final
    torch.save(model.state_dict(), f"{model_checkpoints_path}/{model_final_filename}")
    wandb.save(f"{model_checkpoints_path}/*")
    print("### Models uploaded ###")

    wandb.finish()

    end_time = time.time()
    print(f"Code runs in {end_time - start_time}s")

    # return model, mean_acc


