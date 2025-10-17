# triplet_facenet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
import random
from collections import defaultdict

# -------------------------
# Embedding model (Inception-like)
# -------------------------
class EmbeddingNet(nn.Module):
    def __init__(self, base="inception", emb_size=128, pretrained=True):
        super().__init__()
        if base == "inception":
            # Use torchvision Inception v3 trunk (drop aux logits and final fc)
            backbone = models.inception_v3(pretrained=pretrained, aux_logits=False)
            # remove last fc
            self.features = nn.Sequential(
                *list(backbone.children())[:-1]
            )  # until avgpool
            feat_dim = 2048
        else:
            backbone = models.resnet50(pretrained=pretrained)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            feat_dim = 2048

        self.fc = nn.Linear(feat_dim, emb_size)
        # optionally init fc near zero
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x):
        # features -> flatten -> fc -> l2norm
        f = self.features(x)
        f = f.flatten(1)
        f = self.fc(f)                      
        f = F.normalize(f, p=2, dim=1)  # L2-normalized embedding
        return f

# -------------------------
# PK sampler for batch sampling (P identities, K images each)
# Dataset must allow indexing -> (image, label)
# -------------------------
class PKSampler(Sampler):
    def __init__(self, labels, P, K):
        self.labels = np.array(labels)
        self.P = P
        self.K = K
        self.label2indices = defaultdict(list)
        for idx, l in enumerate(self.labels):
            self.label2indices[l].append(idx)
        self.unique_labels = list(self.label2indices.keys())

    def __iter__(self):
        # yield indices for one epoch; we'll create batches of P*K
        labels = self.unique_labels.copy()
        random.shuffle(labels)
        batch = []
        for _ in range(len(labels) // self.P):
            chosen = random.sample(labels, self.P)
            for lbl in chosen:
                inds = self.label2indices[lbl]
                if len(inds) >= self.K:
                    chosen_inds = random.sample(inds, self.K)
                else:
                    chosen_inds = np.random.choice(inds, self.K, replace=True).tolist()
                batch.extend(chosen_inds)
            yield from batch
            batch = []
        # Note: this simple generator yields many batches; adapt to dataset length

    def __len__(self):
        return len(self.labels) // (self.P * self.K)

# -------------------------
# Pairwise distance matrix (squared)
# -------------------------
def pairwise_distance(embeddings):
    # embeddings: (B, D) L2-normalized
    # squared distances: ||u-v||^2 = ||u||^2 + ||v||^2 - 2 u.v
    # since normalized, ||u||^2 = 1
    dot = torch.matmul(embeddings, embeddings.t())  # (B,B)
    dist = 2 - 2 * dot
    dist.clamp_min_(0.0)
    return dist

# -------------------------
# Online semi-hard triplet mining within batch
# For each anchor-positive pair, pick semi-hard negative:
#   D(a,p) < D(a,n) < D(a,p) + margin
# -------------------------
def batch_all_semi_hard(embeddings, labels, margin=0.2):
    device = embeddings.device
    labels = labels.cpu().numpy()
    dist = pairwise_distance(embeddings).cpu().numpy()
    triplets = []
    B = embeddings.shape[0]
    for i in range(B):
        label_i = labels[i]
        # positives: same label, j != i
        pos_idx = np.where((labels == label_i) & (np.arange(B) != i))[0]
        if len(pos_idx) == 0:
            continue
        for p in pos_idx:
            d_ap = dist[i, p]  
            # find semi-hard negatives n: d_ap < d_an < d_ap + margin
            neg_idx = np.where(labels != label_i)[0]
            d_an = dist[i, neg_idx]
            # candidates
            mask = np.logical_and(d_an > d_ap, d_an < d_ap + margin)
            cand = neg_idx[mask]
            if len(cand) > 0:
                n = np.random.choice(cand)
                triplets.append((i, p, int(n)))
            else:
                # optional: choose hardest negative > d_ap (to avoid collapsing)
                harder = neg_idx[d_an > d_ap]
                if len(harder) > 0:
                    n = harder[np.argmin(dist[i, harder])]
                    triplets.append((i, p, int(n)))
    if len(triplets) == 0:
        return None
    triplets = np.array(triplets)
    return torch.LongTensor(triplets).to(device)

# -------------------------
# Training loop skeleton
# -------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        embeddings = model(imgs)  # (B,128)
        triplets = batch_all_semi_hard(embeddings, labels, margin=loss_fn.margin)
        loss = loss_fn(embeddings, triplets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# -------------------------
# Example usage (fill Dataset implementation & LFW eval)
# -------------------------
if __name__ == "__main__":
    # TODO: implement FaceDataset that returns (img_tensor, label)
    # from your preprocessed crops
    pass

