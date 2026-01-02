# src/model_lstm.py
# Architect Refactored Version: Robust & GPU Optimized

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import copy

# ==========================================
# Core Network Architecture
# ==========================================
class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x: (Batch, Seq, Feat)
        self.lstm.flatten_parameters() 
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ==========================================
# Professional Wrapper (Sklearn-style)
# ==========================================
class LSTMPredictor:
    def __init__(self, 
                 sequence_length=10, 
                 hidden_dim=64, 
                 num_layers=2, 
                 batch_size=1024,      
                 epochs=20,            
                 learning_rate=0.001, 
                 device=None):
        
        self.seq_len = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = learning_rate
        
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.model = None
        self.scaler_X = MinMaxScaler(feature_range=(-1, 1))
        self.scaler_y = MinMaxScaler(feature_range=(-1, 1))
        
        print(f"[System] LSTM Initialized on {self.device} | Batch: {batch_size}")

    def _create_sequences(self, X, y=None):
        """Vectorized sequence creation (Faster than loop)"""
        num_samples = len(X) - self.seq_len
        if num_samples <= 0:
            return np.array([]), np.array([])
            
        # 使用 stride trick 可能更快，但为了稳健性，这里保持列表推导
        xs = [X[i : i+self.seq_len] for i in range(num_samples)]
        
        if y is not None:
            ys = [y[i+self.seq_len] for i in range(num_samples)]
            return np.array(xs), np.array(ys)
        else:
            return np.array(xs), None

    def fit(self, df_train, feature_cols, label_col):
        # 1. Extract & Type Cast
        X = df_train[feature_cols].values.astype(np.float32)
        y = df_train[label_col].values.reshape(-1, 1).astype(np.float32)
        
        # 2. Robust Preprocessing (Winsorization)
        # 去除 1% 的极端异常值，防止 loss 爆炸
        limit_X = np.percentile(np.abs(X), 99, axis=0)
        limit_X[limit_X == 0] = 1.0 # 避免除零
        X = np.clip(X, -limit_X, limit_X)
        
        # 3. Scaling
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y)
        
        # 4. Sequence Generation
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        
        # Move to GPU immediately (Whole dataset)
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=False)
        
        # 5. Model Setup
        input_dim = len(feature_cols)
        self.model = LSTMNet(input_dim, self.hidden_dim, self.num_layers).to(self.device)
        
        # 加入 Weight Decay (L2 正则化) 防止过拟合
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # 6. Training Loop
        self.model.train()
        print(f"[System] Start Training ({self.epochs} epochs)...")
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                preds = self.model(batch_X)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
            
            if (epoch+1) % 5 == 0:
                print(f"Epoch {epoch+1}/{self.epochs} | Loss: {loss.item():.6f}")

    def predict(self, df_test, feature_cols):
        self.model.eval()
        X = df_test[feature_cols].values.astype(np.float32)
        
        # Clip & Scale
        # 注意：这里不用 percentile 计算，直接用 transform，但为了防爆，也可以 clip
        X_scaled = self.scaler_X.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, None)
        
        if len(X_seq) == 0:
            return np.full(len(df_test), np.nan)
            
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        with torch.no_grad():
            preds = self.model(X_tensor)
            preds_np = preds.cpu().numpy()
            
        preds_real = self.scaler_y.inverse_transform(preds_np).flatten()
        
        # Alignment Padding
        padding = np.full(self.seq_len, np.nan)
        return np.concatenate([padding, preds_real])