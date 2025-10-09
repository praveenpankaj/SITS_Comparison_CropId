import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from dataset import SITSData
from msConvSTAR import ConvRNN
# -------------------
# CONFIGURATIONS
# -------------------
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "/root/grapes/best_model.pth"

# Load the test dataset
test_dataset = SITSData()
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=16)
print(len(test_dataset))
# Load the model
model = ConvRNN(input_dim=10, hidden_dim=64).to(DEVICE)
model = model.to(DEVICE)
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
print(checkpoint.keys())
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()  # Set model to evaluation mode

running_cm = np.zeros((2, 2),dtype='int')  # For binary classification
test_loss = 0.0
criterion = torch.nn.BCEWithLogitsLoss()  # Use the same loss function as training

with torch.no_grad():
    for batch in test_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE).unsqueeze(1)  # Shape (batch, 1, 24, 24)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        # print(outputs[0])
        # Convert logits to probabilities
        probs = torch.sigmoid(outputs)

        # Convert probabilities to binary predictions
        preds = (probs > 0.5).float().cpu().numpy().flatten()
        labels = labels.cpu().numpy().flatten()

        # Update confusion matrix
        running_cm += confusion_matrix(labels, preds, labels=[0, 1])
        # print(running_cm)

tn, fp, fn, tp = running_cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Average test loss
test_loss /= len(test_loader)
np.save('/root/working/confusion_matrix.npy',running_cm)
# Print results
print(f"Test Loss: {test_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("Testing complete!")