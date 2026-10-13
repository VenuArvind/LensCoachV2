import torch
import torchvision.transforms as T
import coremltools as ct
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
import ssl

# Fix for macOS SSL certificate error when downloading weights
ssl._create_default_https_context = ssl._create_unverified_context

# 1. Setup Device (Apple Silicon GPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# 2. Dataset Definition
class AADBDataset(Dataset):
    def __init__(self, label_csv, img_dir, transform):
        df = pd.read_csv(label_csv)
        self.img_dir = img_dir
        self.transform = transform
        # Confirmed columns in Dataset.csv (including typos like 'Balacing')
        self.attr_cols = [
            'score', 'BalacingElements', 'ColorHarmony', 'Content', 'DoF', 
            'Light', 'MotionBlur', 'Object', 'Repetition', 'RuleOfThirds', 
            'Symmetry', 'VividColor'
        ]
        # Convert to list of dicts to avoid pandas recursion bugs in __getitem__ (Python 3.13)
        self.data = df[["ImageFile"] + self.attr_cols].to_dict('records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx, retries=0):
        if retries > 3:
            return torch.zeros((3, 224, 224)), torch.zeros(len(self.attr_cols))
            
        item = self.data[idx]
        img_path = os.path.join(self.img_dir, "datasetImages_warp256 2", item['ImageFile'].strip())
        
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            if retries == 0:
                print(f"FAILED to load: {img_path}. Error: {e}")
            return self.__getitem__((idx + 1) % len(self), retries + 1)
            
        img = self.transform(img)
        scores = torch.tensor([item[col] for col in self.attr_cols], dtype=torch.float32)
        return img, scores

# 3. Model Definition (12 outputs)
def get_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(1280, 12) # 11 attributes + 1 score
    )
    for name, param in model.named_parameters():
        if not any(x in name for x in ["features.7", "features.8", "classifier"]):
            param.requires_grad = False
    return model.to(device)

def compute_srcc(predictions, targets):
    """Computes Spearman Rank Correlation Coefficient (SRCC) without external scipy dependency."""
    try:
        from scipy.stats import spearmanr
        srcc, _ = spearmanr(predictions, targets)
        return float(srcc) if not np.isnan(srcc) else 0.0
    except ImportError:
        # Fallback ranking correlation
        pred_ranks = np.argsort(np.argsort(predictions))
        target_ranks = np.argsort(np.argsort(targets))
        n = len(predictions)
        if n <= 1:
            return 0.0
        d_sq = np.sum((pred_ranks - target_ranks) ** 2)
        return float(1.0 - (6.0 * d_sq) / (n * (n**2 - 1)))

def evaluate(model, val_loader, criterion):
    """Evaluates validation loss and SRCC for aesthetic score prediction."""
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            
            # Index 0 is overall aesthetic score
            all_preds.extend(outputs[:, 0].cpu().numpy().tolist())
            all_targets.extend(labels[:, 0].cpu().numpy().tolist())
            
    total_samples = len(all_preds)
    mean_loss = val_loss / max(1, total_samples)
    srcc = compute_srcc(np.array(all_preds), np.array(all_targets))
    return mean_loss, srcc

def export_to_coreml(model, output_path):
    model.eval().cpu()
    example_input = torch.randn(1, 3, 224, 224)
    traced = torch.jit.trace(model, example_input)
    
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image",
            shape=(1, 3, 224, 224),
            scale=1/(255.0 * 0.226),
            bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225]
        )],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL
    )
    mlmodel.author = "LensCoach AI"
    mlmodel.short_description = "AADB Aesthetic Multi-Attribute Predictor"
    mlmodel.save(output_path)
    print(f"✅ Model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    # Transforms
    train_transform = T.Compose([
        T.Resize((240, 240)),
        T.RandomCrop((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset & Train/Val split (lenscoach_v2_plan.md Section 1 & 8)
    full_dataset = AADBDataset(os.path.join(args.data_dir, "Dataset.csv"), args.data_dir, train_transform)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    
    torch.manual_seed(42)
    train_set, val_set = random_split(full_dataset, [train_size, val_size])
    val_set.dataset.transform = val_transform
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    print(f"Dataset split: {train_size} train samples, {val_size} validation samples.")

    # Model, Loss, Optimizer
    model = get_model()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_srcc = -1.0
    print(f"Starting training for {args.epochs} epoch(s)...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (imgs, labels) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if i % 20 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Batch [{i}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        train_loss = running_loss / max(1, len(train_loader))
        val_loss, val_srcc = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch+1} Complete | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val SRCC: {val_srcc:.4f}")
        
        if val_srcc > best_srcc:
            best_srcc = val_srcc
            torch.save(model.state_dict(), "best_framescore_weights.pth")
            print(f"⭐ Saved new best checkpoint with SRCC: {best_srcc:.4f}")

    # Export best model to CoreML
    print(f"Exporting FrameScore model with best SRCC ({best_srcc:.4f}) to CoreML...")
    export_to_coreml(model, "FrameScore.mlpackage")
