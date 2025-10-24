import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def infer_and_compute_loss(model, data_loader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    
    with torch.no_grad():  # No gradients needed for inference
        # Get only the first batch
        images, labels = next(iter(data_loader))
        images, labels = images.to(device), labels.to(device)
        
        # Get features (embeddings)
        features = model(images)  # In eval mode, returns features
        
        # Manually compute loss using A-Softmax
        loss = model.asoftmax(features, labels)
        
        total_loss = loss.item()
        
        # Print embeddings information
        print(f"Sample features shape: {features.shape}")
        print(f"Sample embedding (first sample): {features[0][:10]}...")  # First 10 dims
    
    return total_loss