# Import required libraries
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import densenet201, DenseNet201_Weights
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

# Dataset paths
train_dir = r"C:\Users\Yadhu Gowda\OneDrive\Desktop\chest_xray\train"
val_dir = r"C:\Users\Yadhu Gowda\OneDrive\Desktop\chest_xray\val"
test_dir = r"C:\Users\Yadhu Gowda\OneDrive\Desktop\chest_xray\test"

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, transform=data_transforms['val']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test'])
}

class_names = image_datasets['train'].classes
n_classes = len(class_names)

# Load DenseNet201 with pre-trained weights
model = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)

# Modify classifier
num_ftrs = model.classifier.in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.6),
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, n_classes)
)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Dataloaders
batch_size = 16
dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=0),
    'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False, num_workers=0),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=0)
}

# Evaluate the model on the test set
def evaluate_model(model, dataloader):
    model.eval()
    all_labels = []
    all_preds_prob = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            all_preds_prob.extend(outputs.softmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds_prob)

# Load existing model weights if available
model_weights_path = r"C:\\Users\\svjos\\OneDrive\\Desktop\\densenet201_model.pth"
if os.path.exists(model_weights_path):
    print("Loading existing model weights...")
    model.load_state_dict(torch.load(model_weights_path))
else:
    print("Model weights not found. Using pre-trained DenseNet201 weights without further training.")

# Evaluate model and calculate metrics
all_labels, all_preds_prob = evaluate_model(model, dataloaders['test'])

# Check and preprocess labels and predictions for ROC and AUC
if n_classes > 2:
    # One-hot encode the labels for multiclass
    all_labels_bin = label_binarize(all_labels, classes=range(n_classes))
else:
    # For binary classification
    all_labels_bin = all_labels
    all_preds_prob = all_preds_prob[:, 1]  # Take the positive class probability

# Compute ROC and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    if n_classes > 2:
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_preds_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        fpr[i], tpr[i], _ = roc_curve(all_labels_bin, all_preds_prob)
        roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    if n_classes > 2:
        plt.plot(fpr[i], tpr[i], label=f"Class {class_names[i]} (AUC = {roc_auc[i]:.2f})")
    else:
        plt.plot(fpr[i], tpr[i], label=f"Binary Classification (AUC = {roc_auc[i]:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.title("ROC Curves")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="best")
plt.grid()
plt.show()

# Print mean AUC for multiclass
if n_classes > 2:
    mean_auc = np.mean(list(roc_auc.values()))
    print(f"Mean AUC: {mean_auc:.2f}")
