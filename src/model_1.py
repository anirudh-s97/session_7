import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Block 1: Stronger initial feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 15, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Conv2d(15, 15, kernel_size=3, padding=0), 
            nn.ReLU(), 
            nn.BatchNorm2d(15),
            nn.Conv2d(15, 15, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.MaxPool2d(2, 2), 
            )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(15, 15, kernel_size=3), 
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Conv2d(15, 15, kernel_size=3),   # 2,304 + 16 = 2,320 params
            nn.ReLU(),
            nn.BatchNorm2d(15),
            nn.Dropout(0.05),
            nn.Conv2d(15, 15, kernel_size=3),   # 2,304 + 16 = 2,320 params
            nn.ReLU(),
            nn.BatchNorm2d(15),                                                       
        )
        
        
        # Block 3: Efficient feature refinement
        self.block3 = nn.Sequential(
            nn.Conv2d(15, 12, kernel_size=3),   # 3,456 + 24 = 3,480 params   
            nn.ReLU(),
            nn.BatchNorm2d(12),
           
        )

        self.gap =  self.gap = nn.Sequential(
            #nn.AvgPool2d(kernel_size=7)
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(12, 10, kernel_size=1, padding=0)   # 3,456 + 24 = 3,480 params
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.block4(x)
        x = x.view(-1, 10 * 1 * 1)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    print("The total number of parameters from this architecture is: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)), flush=True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = MNISTClassifier()
    
    print("\nParameter count per layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
    
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")