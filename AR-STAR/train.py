import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
import numpy as np

from dataset import SITSData
from msConvSTAR import ConvRNN

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/root/grapes/checkpoints/best_model.pth"

train_dataset = SITSData(mode="train")
val_dataset = SITSData(mode='test')
print(len(train_dataset),len(val_dataset))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=24)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

# -------------------
# MODEL SETUP
# -------------------
model = ConvRNN(input_dim=10, hidden_dim=64).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

start_epoch = 0
# if os.path.exists(CHECKPOINT_PATH):
#     checkpoint = torch.load(CHECKPOINT_PATH)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     start_epoch = checkpoint["epoch"] + 1
#     print(f"Resuming training from epoch {start_epoch}...")


writer = SummaryWriter("/root/tracking/binary_classification")
best_f1, best_acc = 0.0, 0.0

# training loop
for epoch in range(EPOCHS):
    model.train()  # Enable training mode
    train_loss = 0.0
    print(f"Epoch {epoch+1}\n")
    for i,batch in enumerate(train_loader):
        inputs, labels = batch
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)  # Reshape labels to (batch, 1, 24, 24)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    #  VALIDATION LOOP
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        running_cm = np.zeros((2, 2)) 
        for j,batch in enumerate(val_loader):
            inputs, labels = batch
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)
            outputs = model(inputs)
            # print(torch.unique(labels))
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs)

            # Convert probabilities to binary predictions
            preds = (probs > 0.5).float().cpu().numpy().flatten()
            labels = labels.cpu().numpy().flatten()
            running_cm += confusion_matrix(labels,preds,labels=[0,1])
    
    # print(running_cm)
    tn, fp, fn, tp = running_cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Average losses
    train_loss /= i
    val_loss /= j

    # Print progress
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f},'\n', Precision: {precision:.4f}, '\n',Recall: {recall:.4f}, '\n',F1-Score: {f1:.4f}")

    # Log to TensorBoard
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Metrics/Accuracy", accuracy, epoch)
    writer.add_scalar("Metrics/Precision", precision, epoch)
    writer.add_scalar("Metrics/Recall", recall, epoch)
    writer.add_scalar("Metrics/F1-Score", f1, epoch)

    # saving the model based on F1 score or accuracy
    if best_f1 < f1 or best_acc < accuracy:
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, CHECKPOINT_PATH)

# Close TensorBoard writer
writer.close()

print("Training complete! Run `tensorboard --logdir=runs` to visualize.")
