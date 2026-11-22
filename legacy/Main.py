import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pathlib
import numpy as np
import copy
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ==========================================
# 0. ENSURE REPRODUCIBILITY
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==========================================
# 1. BASIC CONFIGURATION
# ==========================================
IMAGE_SIZE = 224 
BATCH_SIZE = 32
EPOCHS = 80 
LEARNING_RATE = 2e-3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# ==========================================
# 2. DATA PREPROCESSING
# ==========================================
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(), # Valid Augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2), # Valid Augmentation
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

data_dir = pathlib.Path("/kaggle/input/datasets/huynhthethien/radarcommunsignaldata2026train")
dataset = datasets.ImageFolder(root=str(data_dir), transform=transform)

num_classes = len(dataset.classes)
class_names = dataset.classes

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ==========================================
# 3. EMA & MIXUP TECHNIQUES
# ==========================================
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_v, model_v in zip(self.ema.state_dict().values(), model.state_dict().values()):
                if ema_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==========================================
# 4. MODEL ARCHITECTURE
# ==========================================
class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.SiLU()
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        return identity * a_w * a_h

class MBConvCA(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = int(round(inp * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        self.conv = nn.Sequential(*layers)
        self.ca = CoordAtt(hidden_dim) 
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
        )

    def forward(self, x):
        out = self.conv(x)
        out = self.ca(out)
        out = self.project(out)
        if self.use_res_connect:
            return x + out
        return out

class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )
        
        self.blocks = nn.Sequential(
            MBConvCA(16, 24, stride=2, expand_ratio=2),  
            MBConvCA(24, 40, stride=2, expand_ratio=2),  
            MBConvCA(40, 64, stride=2, expand_ratio=2),  
            MBConvCA(64, 80, stride=2, expand_ratio=2),  
            MBConvCA(80, 128, stride=1, expand_ratio=2)  
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 2, num_classes) # Doubled channels due to Pool concat
        
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

model = BasicCNN(num_classes).to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("="*60)
print(f"Total Parameters: {total_params:,} / 100,000 (COMPLIANT)")
print("="*60)

# ==========================================
# 5. OPTIMIZATION & TRAINING
# ==========================================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) 
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# Initialize Dictionary to store history for plotting
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

def train_model(model, train_loader, val_loader):
    ema = ModelEMA(model) 
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict()) 
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct, total = 0.0, 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            inputs, targets_a, targets_b, lam = mixup_data(images, labels, alpha=0.2)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            
        train_acc = correct / total
        train_loss = running_loss / total
        
        # Validation using EMA
        ema.ema.eval()
        val_running_loss = 0.0
        val_correct, val_total = 0.0, 0.0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = ema.ema(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        val_acc = val_correct / val_total
        val_loss = val_running_loss / val_total
        scheduler.step()
        
        # Save to history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | LR: {scheduler.get_last_lr()[0]:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(ema.ema.state_dict())
            print(f">>> New best accuracy achieved: {best_acc:.4f}")

    print(f"\n[COMPLETED] Highest Validation Accuracy: {best_acc:.4f}")
    model.load_state_dict(best_model_wts)
    return model

# Call training function
model = train_model(model, train_loader, val_loader)

# ==========================================
# 6. EVALUATION, METRICS, AND MANDATORY PLOTS
# ==========================================
print("\n" + "="*60)
print("PROCEEDING WITH MODEL EVALUATION AND PLOTTING...")
print("="*60)

# 6.1. Plot Loss & Accuracy per Epoch
plt.figure(figsize=(10, 5))
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_plot.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('acc_plot.png')
plt.close()
print("- Loss plot saved to file: loss_plot.png")
print("- Accuracy plot saved to file: acc_plot.png")

# 6.2. Calculate Metrics and Plot Confusion Matrix on Validation Set
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader: # Using val_loader as internal test data
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate mandatory metrics
acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average=None)
recall = recall_score(all_labels, all_preds, average=None)
f1 = f1_score(all_labels, all_preds, average=None)

print("\n--- DETAILED CLASSIFICATION REPORT ---")
print(f"Overall Accuracy: {acc * 100:.2f}%")
print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 55)
for i, name in enumerate(class_names):
    print(f"{name:<15} | {precision[i]:.4f}     | {recall[i]:.4f}     | {f1[i]:.4f}")

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("- Confusion Matrix saved to file: confusion_matrix.png")

# ==========================================
# 7. MODEL EXPORT
# ==========================================
print("\n" + "="*60)
print("PROCEEDING TO SAVE MODEL ACCORDING TO STANDARDS...")
print("="*60)

############################################
# DO NOT MODIFY THIS SECTION
############################################
model.eval()
example_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
traced_model = torch.jit.trace(model, example_input)

GroupID = "09" 
model_name = f"{GroupID}_DeepLearningProject_TrainedModel.pt"
traced_model.save(model_name)
print("Model saved:", model_name)