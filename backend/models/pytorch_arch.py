"""Arquitecturas PyTorch: GRU apilada con atención e híbrido CNN multiescala + BiGRU + atención."""
import torch
import torch.nn as nn
 
 
class Attention(nn.Module):
    """Atención temporal: pondera los pasos de la ventana según su relevancia."""
    def __init__(self, d):
        super().__init__()
        self.w = nn.Linear(d, 1)
 
    def forward(self, h):
        return (torch.softmax(self.w(h), 1) * h).sum(1)
 
 
class MultiScaleConv(nn.Module):
    """Convoluciones paralelas (kernels 3, 5 y 7) que capturan patrones de distinta duración."""
    def __init__(self, in_dim, ch=16):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(in_dim, ch, k, padding=k // 2) for k in (3, 5, 7)])
        self.bn, self.act = nn.BatchNorm1d(3 * ch), nn.GELU()
 
    def forward(self, x):  # x: (B, L, F)
        x = x.transpose(1, 2)
        return self.act(self.bn(torch.cat([c(x) for c in self.convs], 1))).transpose(1, 2)
 
 
def _head(d):
    return nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, 1))
 
 
class GRUForecaster(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=2, batch_first=True, dropout=0.2)
        self.att, self.head = Attention(hid), _head(2 * hid)
 
    def forward(self, x):
        o, _ = self.gru(x)
        return self.head(torch.cat([o[:, -1], self.att(o)], 1))
 
 
class LSTMForecaster(nn.Module):
    """Igual que GRUForecaster pero con celdas LSTM (permite comparar ambos tipos de RNN)."""
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.rnn = nn.LSTM(in_dim, hid, num_layers=2, batch_first=True, dropout=0.2)
        self.att, self.head = Attention(hid), _head(2 * hid)
 
    def forward(self, x):
        o, _ = self.rnn(x)
        return self.head(torch.cat([o[:, -1], self.att(o)], 1))
 
 
class CNNGRUForecaster(nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.conv = MultiScaleConv(in_dim, 16)
        self.gru = nn.GRU(48, hid, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.att, self.head = Attention(2 * hid), _head(4 * hid)
 
    def forward(self, x):
        o, _ = self.gru(self.conv(x))
        return self.head(torch.cat([o[:, -1], self.att(o)], 1))
 
 
def build(kind, in_dim):
    if kind == "gru":
        return GRUForecaster(in_dim)
    if kind == "lstm":
        return LSTMForecaster(in_dim)
    return CNNGRUForecaster(in_dim)
