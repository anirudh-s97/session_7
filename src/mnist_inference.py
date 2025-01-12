import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import numpy as np
import albumentations as A
from model_3 import MNISTClassifier

def create_folders():
    """Create folders to store misclassified images"""
    folders = ['misclassified_train', 'misclassified_test']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

def save_misclassified_images(images, predictions, labels, indices, folder):
    """Save misclassified images to specified folder"""
    for img, pred, label, idx in zip(images, predictions, labels, indices):
        # Convert tensor to PIL Image
        img = img.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC format
        img = (img * 255).astype(np.uint8)
        if img.shape[-1] == 1:  # If single channel, convert to RGB
            img = np.repeat(img, 3, axis=-1)
        img = Image.fromarray(img)
        
        # Save image with informative filename
        filename = f"idx_{idx}_pred_{pred}_true_{label}.png"
        img.save(os.path.join(folder, filename))

def evaluate_model(model, dataloader, device, folder):
    """Evaluate model and save misclassified images"""
    model.eval()
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    misclassified_indices = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            
            # Find misclassified indices
            incorrect_mask = pred.ne(target)
            if incorrect_mask.any():
                misclassified_images.extend(data[incorrect_mask])
                misclassified_preds.extend(pred[incorrect_mask].cpu().numpy())
                misclassified_labels.extend(target[incorrect_mask].cpu().numpy())
                misclassified_indices.extend(
                    [batch_idx * dataloader.batch_size + i for i, x in enumerate(incorrect_mask) if x]
                )
    
    # Save misclassified images
    save_misclassified_images(
        misclassified_images,
        misclassified_preds,
        misclassified_labels,
        misclassified_indices,
        folder
    )
    
    return len(misclassified_images)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create folders for saving images
    create_folders()

    from albumentations.pytorch import ToTensorV2
    class AlbumentationsTransform:
        
        def __init__(self, transform):
            self.transform = transform

        def __call__(self, img):
            # Convert PIL Image to numpy array
            img = np.array(img)
            # Apply Albumentations transform with named argument
            transformed = self.transform(image=img)
            return transformed["image"]
        
    
    train_transform = transforms.Compose([
        transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
        AlbumentationsTransform(A.CoarseDropout(num_holes_range=(60, 90), hole_height_range=(2,2), hole_width_range=(2,2), p=1.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

        # Load MNIST datasets
    train_dataset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load your model here
    # Example: Assuming you have a simple CNN model
    
    model = MNISTClassifier()
    state_dict = torch.load(r'C:\Users\aniru\Downloads\session_6\src\mnist_classifier.pth')
    model.load_state_dict(state_dict)
    model = model.to(device)
    # model = torch.load_state_dict(r'C:\Users\aniru\Downloads\session_6\src\mnist_classifier.pth')  # Replace with your model path
    # model = model.to("cpu")
    
    # Evaluate and save misclassified images
    print("Evaluating training set...")
    train_misclassified = evaluate_model(
        model, train_loader, device, 'misclassified_train'
    )
    
    print("Evaluating test set...")
    test_misclassified = evaluate_model(
        model, test_loader, device, 'misclassified_test'
    )
    
    print(f"Found {train_misclassified} misclassified training images")
    print(f"Found {test_misclassified} misclassified test images")

if __name__ == "__main__":
    main()
