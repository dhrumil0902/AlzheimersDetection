import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os

N_SAMPLES = 325


class CFCDataset(Dataset):
    def __init__(self):
        self.file_paths = glob.glob("data/cfc/cfc_*_[ca][nd].npy")
        self.labels = [1 if 'ad' in path else 0 for path in self.file_paths]
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load and normalize data if not already normalized
        data = np.load(self.file_paths[idx]).astype(np.float32)
        # Ensure shape is 6x40x30
        assert data.shape == (6, 40, 30)
        return torch.from_numpy(data), self.labels[idx]

class CFCClassifier(nn.Module):
    def __init__(self):
        super(CFCClassifier, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.2)
        )
        
        # Temporal layers (using LSTM)
        self.temporal_size = 64 * 6 * 30  # After conv layers
        self.lstm = nn.LSTM(
            input_size=self.temporal_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Self-attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.2
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for conv layers (add channel dimension)
        x = x.view(batch_size * 6, 1, 40, 30)
        
        # Apply conv layers
        x = self.conv_layers(x)
        
        # Reshape for temporal processing
        x = x.view(batch_size, 6, -1)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Apply self-attention
        # Reshape for attention (seq_len, batch, features)
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        
        # Take the last temporal step
        x = x[-1]
        
        # Apply fully connected layers
        x = self.fc_layers(x)
        
        return x.squeeze()

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation phase
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.float().to(device)
                
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                
                predicted = (outputs > 0.5).float()
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss/len(train_loader):.4f}')
        print(f'Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Val Accuracy: {val_accuracy:.2f}%')
        
        scheduler.step(val_loss)

def main():
    # Create dataset
    dataset = CFCDataset()
    
    # Split into train/val sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize and train model
    model = CFCClassifier()
    train_model(model, train_loader, val_loader)

if __name__ == "__main__":
    main()