from facenet_pytorch import MTCNN
import torch
from torchvision import transforms
from PIL import Image
import os

def preprocess_dataset(original_path, aligned_path, image_height=112, image_width=96):
    """
    Preprocess the dataset by aligning faces using MTCNN and saving to a new directory.
    
    Args:
        original_path: Path to the original dataset directory.
        aligned_path: Path to save the aligned images.
        image_height: Height of the output aligned images (default 112).
        image_width: Width of the output aligned images (default 96).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], 
                   factor=0.709, post_process=False, device=device)
    
    resize_transform = transforms.Resize((image_height, image_width))

    os.makedirs(aligned_path, exist_ok=True)
    
    for root, dirs, files in os.walk(original_path):
        rel_path = os.path.relpath(root, original_path)
        new_root = os.path.join(aligned_path, rel_path)
        os.makedirs(new_root, exist_ok=True)
        
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path).convert('RGB')
                    aligned = mtcnn(img)
                    if aligned is not None:
                        aligned = resize_transform(aligned)
                        aligned = aligned.permute(1, 2, 0).byte().numpy()  # Convert to numpy array for saving
                        aligned_img = Image.fromarray(aligned)
                        aligned_img.save(os.path.join(new_root, file))
                    else:
                        print(f"No face detected in {img_path}, skipping alignment.")
                        # Optionally copy original: shutil.copy(img_path, os.path.join(new_root, file))
                except Exception as e:
                    raise Exception(f"Error processing {img_path}: {e}")
