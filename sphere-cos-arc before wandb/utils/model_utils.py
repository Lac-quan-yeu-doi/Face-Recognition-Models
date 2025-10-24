import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from .config import *
from .dataset import LFWPairDataset

# Custom Collate Function for Handling None Returns
def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.dataloader.default_collate(batch)

# Training Function
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
    return avg_loss

# Evaluation Function (Classification Accuracy)
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
            batch_size = labels.size(0)
            theta = torch.acos(cos_theta.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
            cos_theta_m = torch.cos(theta + model.arcface.m * nn.functional.one_hot(labels, num_classes=model.arcface.num_classes))
            scaled_logits = model.arcface.s * torch.where(nn.functional.one_hot(labels, num_classes=model.arcface.num_classes).bool(), cos_theta_m, cos_theta)
            _, predicted = torch.max(scaled_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# LFW Verification Evaluation (Single Set or Dev Sets)
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

# LFW 10-Fold Cross-Validation Evaluation (Updated for 10 300 300 header)
def evaluate_lfw_10fold(model, pairs_file, batch_size, root_dir, transform, device, threshold=0.5):
    model.eval()
    fold_accuracies = []
    
    # Parse pairs.txt for 10 folds with 300 same + 300 different pairs each
    with open(pairs_file, 'r') as f:
        lines = f.readlines()
        header = lines[0].strip().split()
        num_folds, num_same, num_diff = map(int, header)  # Expecting 10 300 300
        pairs_per_fold = num_same + num_diff  # 600 pairs per fold
        lines = lines[1:]  # Skip header
    
    for fold in range(num_folds):
        # Extract pairs for the current fold
        start_idx = fold * pairs_per_fold
        end_idx = (fold + 1) * pairs_per_fold
        fold_pairs = []
        same_count = 0
        diff_count = 0
        for line in lines[start_idx:end_idx]:
            parts = line.strip().split()
            if len(parts) == 3:  # Same identity
                name, img1, img2 = parts
                fold_pairs.append((f"{name}/{name}_{img1.zfill(4)}.jpg", 
                                 f"{name}/{name}_{img2.zfill(4)}.jpg", 1))
                same_count += 1
            elif len(parts) == 4:  # Different identities
                name1, img1, name2, img2 = parts
                fold_pairs.append((f"{name1}/{name1}_{img1.zfill(4)}.jpg", 
                                 f"{name2}/{name2}_{img2.zfill(4)}.jpg", 0))
                diff_count += 1
            else:
                print(f"Warning: Invalid line format in fold {fold + 1}: {line.strip()}")
        # Verify pair counts
        if same_count != num_same or diff_count != num_diff:
            print(f"Warning: Fold {fold + 1} has {same_count} same pairs and {diff_count} different pairs, expected {num_same} same and {num_diff} different")
        
        # Create a temporary dataset for this fold
        temp_dataset = LFWPairDataset(root_dir, pairs_file, transform)
        temp_dataset.pairs = fold_pairs  # Override pairs with fold-specific pairs
        accuracy = evaluate_lfw_verification(model, temp_dataset, batch_size, device, threshold)
        fold_accuracies.append(accuracy)
        print(f'Fold {fold + 1} Verification Accuracy: {accuracy:.2f}% (Same: {same_count}, Different: {diff_count})')
    
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    return mean_accuracy, std_accuracy

# Threshold Tuning Function
def tune_threshold(model, pairs_dataset, batch_size, device, thresholds=np.arange(0.1, 0.9, 0.05)):
    best_threshold = 0.5
    best_accuracy = 0
    print("Tuning verification threshold...")
    for thresh in thresholds:
        accuracy = evaluate_lfw_verification(model, pairs_dataset, batch_size, device, thresh)
        print(f'Threshold {thresh:.2f}: Verification Accuracy {accuracy:.2f}%')
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = thresh
    return best_threshold, best_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--batch-size', type=int, default=512, 
                       help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='number of epochs to train (default: 30)')
    parser.add_argument('--lr', type=float, default=0.01, 
                       help='learning rate (default: 0.01)')
    parser.add_argument('--model-save-path', type=str, default=f'{WORKING_PATH}/models',
                       help='path to save models (default: WORKING_PATH/models)')
    return parser.parse_args()
