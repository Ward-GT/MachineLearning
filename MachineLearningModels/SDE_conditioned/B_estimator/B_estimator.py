from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import torch.nn as nn
import numpy as np
import os
import re
import pandas as pd
import torch

def calculate_mape(y_true, y_pred):
    # Convert to numpy arrays if not already
    if y_true.is_cuda:
        y_true = y_true.cpu()

    if y_pred.is_cuda:
        y_pred = y_pred.cpu()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero by masking zero elements
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]

    # Compute MAPE
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    maxpe = np.max(np.abs((y_true - y_pred) / y_true)) * 100
    return mape, maxpe

class TensorStandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, tensor):
        self.mean = tensor.mean(0, keepdim=True)
        self.std = tensor.std(0, keepdim=True)
        self.std[self.std == 0] = 1  # Prevent division by zero

    def transform(self, tensor):
        return (tensor - self.mean) / self.std

    def fit_transform(self, tensor):
        self.fit(tensor)
        return self.transform(tensor)

    def inverse_transform(self, tensor):
        return tensor * self.std + self.mean

def load_data_to_tensor(input_dir):
    df_bmax = pd.read_csv(input_dir, on_bad_lines='skip')
    df_bmax = df_bmax.drop('freq', axis=1)
    input_data = df_bmax.iloc[:, :9].to_numpy()
    output_data = df_bmax.iloc[:, 9].to_numpy()
    input_tensors = torch.tensor(input_data, dtype=torch.float32)
    output_tensors = torch.tensor(output_data, dtype=torch.float32).reshape(-1, 1)
    return input_tensors, output_tensors

class RegressionDataset(Dataset):
    def __init__(self, input_dir):
        self.input_dir = input_dir
        self.input_tensors, self.output_tensors = load_data_to_tensor(input_dir)
        self.InputScaler = TensorStandardScaler()
        self.input_tensors = self.InputScaler.fit_transform(self.input_tensors) # Normalize the input with zero mean and variance of 1
        self.output_tensors = self.output_tensors * 1000 # Multiply output by 1000 because of small values

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]

class MLPRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super().__init__()

        # Build layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.Dropout(0.1)
            ])
            prev_dim = dim

        # Output layer
        layers.append(nn.Linear(prev_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def get_dataloaders(input_dir, batch_size):
    # Create datasets and dataloaders
    dataset = RegressionDataset(input_dir)

    train_size = 0.7
    val_size = 0.3

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size
    )
    return train_loader, val_loader

def train_model(input_dir, model_params={}):
    # Default parameters
    default_model_params = {
        'hidden_dims': [128, 64, 32, 16],
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 2000
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Update defaults with provided parameters
    params = {**default_model_params, **model_params}

    train_loader, val_loader = get_dataloaders(input_dir, params['batch_size'])

    # Initialize model
    model = MLPRegressor(
        input_dim=9,
        hidden_dims=params['hidden_dims']
    )

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(params['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                mape, _ = calculate_mape(y_pred, y_batch)
                val_loss += mape

        # Print progress
        if (epoch + 1) % 10 == 0:
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            print(f'Epoch [{epoch + 1}/{params["epochs"]}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()

    # Load best model
    model.load_state_dict(best_model)

    return model, train_loader, val_loader

if __name__ == "__main__":
    input_dir = r"C:\Users\tabor\Documents\Studie\Bachelor\Jaar 4\BEP\Data\figure_B_maxrange_5000\Bmax_maxrange_5000.csv"
    model, train_loader, val_loader = train_model(input_dir)