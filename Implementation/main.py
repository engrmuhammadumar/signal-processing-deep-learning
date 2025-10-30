import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    writer = SummaryWriter('runs/pump_diagnosis')
    
    best_val_acc = 0
    
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.NUM_EPOCHS}'):
            time_series = batch['time_series'].to(device)
            scalogram = batch['scalogram'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(time_series, scalogram)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                time_series = batch['time_series'].to(device)
                scalogram = batch['scal
