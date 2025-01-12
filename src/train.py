import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import sys
import albumentations as A

model_id = sys.argv[1]


if model_id == "model_1":
    from model_1 import MNISTClassifier, count_parameters

elif model_id == "model_2":
    from model_2 import MNISTClassifier, count_parameters  # Import your model

elif model_id == "model_3":
    from cutout import Cutout
    from model_3 import MNISTClassifier, count_parameters  # Import your model

else:
    raise("Invalid Model id to train")


def train_model(model, train_loader, test_loader, criterion, optimizer, epochs=15, device='cuda'):
    # Initialize lists to store metrics
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    # Move model to device
    model = model.to(device)

    num_parameters = count_parameters(model)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Evaluation phase
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100. * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch: {epoch+1}/{epochs}')
        print(f'Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%\n')
    
    return train_losses, test_losses, train_accuracies, test_accuracies

def plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
        ax1.annotate(f'{train_loss:.3f}', (epochs[i], train_losses[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax1.annotate(f'{test_loss:.3f}', (epochs[i], test_losses[i]), textcoords="offset points", xytext=(0,-15), ha='center')
    ax1.set_title('Training and Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    ax2.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    for i, (train_acc, test_acc) in enumerate(zip(train_accuracies, test_accuracies)):
        ax2.annotate(f'{train_acc:.1f}%', (epochs[i], train_accuracies[i]), textcoords="offset points", xytext=(0,10), ha='center')
        ax2.annotate(f'{test_acc:.1f}%', (epochs[i], test_accuracies[i]), textcoords="offset points", xytext=(0,-15), ha='center')
    ax2.set_title('Training and Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if model_id == "model_1":
        train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
        
        test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    
    elif model_id == "model_2":
    
        # Define transforms
        train_transform = transforms.Compose([
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    else:
        # Define transforms
        from albumentations.pytorch import ToTensorV2
        class AlbumentationsTransform:
            
            def __init__(self, transform):
                self.transform = transform

            def __call__(self, img):
                # Convert PIL Image to numpy array
                img = np.array(img)
                # Apply Albumentations transform with named argument
                transformed = self.transform(image=img, mask=img)
                return transformed["image"]
            
        
        train_transform = transforms.Compose([
            transforms.RandomRotation((-7.0, 7.0), fill=(1,)),
            AlbumentationsTransform(A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(4, 5), hole_width_range=(5, 5), p=1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    
       
        # transform = A.Compose([
        #     A.Rotate(limit=(-7.0, 7.0), border_mode=0, p=1.0, value=1),
        #     A.CoarseDropout(max_holes=1, hole_height_range=(8,8), hole_width_range=(5,5), p=1.0),
        #     A.Normalize((0.1307,), (0.3081,)),
        #     ToTensorV2(),
        # ])
        #train_transform = AlbumentationsTransform(transform)



        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)

    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    #Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Initialize model, criterion, and optimizer
    model = MNISTClassifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model and get metrics
    train_losses, test_losses, train_accuracies, test_accuracies = train_model(
        model, train_loader, test_loader, criterion, optimizer, epochs=15, device=device
    )
    
    # Plot and save metrics
    plot_metrics(train_losses, test_losses, train_accuracies, test_accuracies)
    
    # Save the trained model
    torch.save(model.state_dict(), 'mnist_classifier.pth')

if __name__ == '__main__':
    main()