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
import torch
import torch.nn as nn
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
from utils.dataset import CASIAwebfaceDataset, LFWPairDataset
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
    """
    Perform 10-fold cross-validation on LFW dataset using pairs.csv.
    Args:
        model: Trained model for evaluation.
        pairs_file (str): Path to pairs.csv file.
        batch_size (int): Batch size for evaluation.
        root_dir (str): Path to dataset root (e.g., aligned_lfw_path).
        transform: torchvision transforms for preprocessing.
        device: Device to run the model (cuda or cpu).
        threshold (float): Threshold for verification.
    Returns:
        mean_accuracy (float): Mean accuracy across 10 folds.
        std_accuracy (float): Standard deviation of accuracies.
    """
    model.eval()
    fold_accuracies = []
    num_folds = 10

    # Load all pairs from pairs.csv
    all_pairs = []
    with open(pairs_file, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)  # Skip header (e.g., name,imagenum1,imagenum2,)
        for row in reader:
            if len(row) == 4 and (row[-1] in ['', ' ']):
                row = row[:3]
            if len(row) == 3:  # Matched pair: name,imagenum1,imagenum2
                name, img1, img2 = row
                img1_path = f"{name}/{name}_{img1.zfill(4)}.jpg"
                img2_path = f"{name}/{name}_{img2.zfill(4)}.jpg"
                all_pairs.append((img1_path, img2_path, 1))
            elif len(row) == 4:  # Mismatched pair: name1,imagenum1,name2,imagenum2
                name1, img1, name2, img2 = row
                img1_path = f"{name1}/{name1}_{img1.zfill(4)}.jpg"
                img2_path = f"{name2}/{name2}_{img2.zfill(4)}.jpg"
                all_pairs.append((img1_path, img2_path, 0))
            else:
                print(f"Skipping invalid row in {pairs_file}: {row}")

    # Separate matched and mismatched pairs
    matched_pairs = [p for p in all_pairs if p[2] == 1]
    mismatched_pairs = [p for p in all_pairs if p[2] == 0]

    # Ensure equal number of matched and mismatched pairs per fold
    pairs_per_fold = min(len(matched_pairs), len(mismatched_pairs)) // num_folds
    if pairs_per_fold == 0:
        raise ValueError("Not enough pairs to distribute across 10 folds")

    # Shuffle pairs to randomize fold assignment
    np.random.shuffle(matched_pairs)
    np.random.shuffle(mismatched_pairs)

    # Create 10 folds with stratified sampling
    for fold in range(num_folds):
        start_idx = fold * pairs_per_fold
        end_idx = (fold + 1) * pairs_per_fold

        # Select pairs for this fold
        fold_matched = matched_pairs[start_idx:end_idx]
        fold_mismatched = mismatched_pairs[start_idx:end_idx]
        fold_pairs = fold_matched + fold_mismatched

        # Ensure non-empty fold
        if not fold_pairs:
            print(f"Warning: Fold {fold + 1} is empty, skipping")
            continue

        # Create dataset for this fold
        temp_dataset = LFWPairDataset(root_dir=root_dir, pairs_files=None, transform=transform)
        temp_dataset.pairs=fold_pairs
        # Evaluate accuracy for this fold
        accuracy = evaluate_lfw_verification(model, temp_dataset, batch_size, device, threshold)
        fold_accuracies.append(accuracy)
        print(f'Fold {fold + 1} Verification Accuracy: {accuracy:.2f}%')

    if not fold_accuracies:
        raise ValueError("No valid folds were evaluated")

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
        print(f'Threshold {thresh:.3f}: Verification Accuracy {accuracy:.3f}%')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    return best_threshold, best_accuracy, threshold_list, accuracy_list

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-bs', type=int, default=512)
    parser.add_argument('--epochs', '-e', type=int, default=30)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.1)
    parser.add_argument('--backbone', '-bb', type=str, default='resnet18')
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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ])

    train_dataset = CASIAwebfaceDataset(root_dir=f"{dataset_path}/CASIA-WebFace", split='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, collate_fn=custom_collate_fn)

    # === Model, Opt, Scheduler ===
    model = model_class(num_classes=num_classes, backbone=args.backbone).to(device)
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

    # === Evaluation ===
    # Load best model
    best_path = f"{model_checkpoints_path}/{model_name}_min_loss.pth"
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

    # Threshold tuning
    aligned_lfw_path = f'{dataset_path}/LFW'
    match_train_file = f'{dataset_path}/LFW/matchpairsDevTrain.csv'
    mismatch_train_file = f'{dataset_path}/LFW/mismatchpairsDevTrain.csv'
    match_test_file = f'{dataset_path}/LFW/matchpairsDevTest.csv'
    mismatch_test_file = f'{dataset_path}/LFW/mismatchpairsDevTest.csv'
    pairs_all_file = f'{dataset_path}/LFW/pairs.csv'

    # --- Combine DevTrain + DevTest for threshold tuning ---
    train_pairs_dataset_match = LFWPairDataset(
        root_dir=aligned_lfw_path,
        pairs_files=match_train_file,
        transform=test_transform
    )
    train_pairs_dataset_mismatch = LFWPairDataset(
        root_dir=aligned_lfw_path,
        pairs_files=mismatch_train_file,
        transform=test_transform
    )
    test_pairs_dataset_match = LFWPairDataset(
        root_dir=aligned_lfw_path,
        pairs_files=match_test_file,
        transform=test_transform
    )
    test_pairs_dataset_mismatch = LFWPairDataset(
        root_dir=aligned_lfw_path,
        pairs_files=mismatch_test_file,
        transform=test_transform
    )
    # Combine all pairs for threshold tuning
    combined = ConcatDataset([
        train_pairs_dataset_match,
        train_pairs_dataset_mismatch,
        test_pairs_dataset_match,
        test_pairs_dataset_mismatch
    ])


    best_thresh, best_acc, t_list, a_list = tune_threshold(model, combined, 512, device, thresholds=np.arange(-0.50, 0.50, 0.002))
    print(f"Best Threshold: {best_thresh:.3f} → {best_acc:.2f}%")

    # 10-fold LFW
    mean_acc, std_acc = evaluate_lfw_10fold(
        model, f"{dataset_path}/LFW/pairs.csv", 512,
        f"{dataset_path}/LFW", test_transform, device, best_thresh
    )
    print(f"LFW 10-Fold: {mean_acc:.2f}% ± {std_acc:.2f}%")

    # Save final
    torch.save(model.state_dict(), f"{model_checkpoints_path}/{model_final_filename}")
    wandb.save(f"{model_checkpoints_path}/*")

    end_time = time.time()
    print(f"Code runs in {end_time - start_time}s")
    
    wandb.finish()

    return model, best_thresh, mean_acc


