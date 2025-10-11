import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# Assuming SBERT and SBERTClassification classes are defined in the model.py file
from model import SBERT, SBERTClassification

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    return (po - pe) / (1 - pe)


def average_accuracy(matrix):
    correct = np.diag(matrix)
    all = matrix.sum(axis=0)
    accuracy = correct / all
    aa = np.average(accuracy)
    return aa


class SBERTFineTuner:
    def __init__(self, sbert: SBERT, num_classes: int,
                 train_dataloader: DataLoader, valid_dataloader: DataLoader,
                 lr: float = 1e-4, with_cuda: bool = True,
                 cuda_devices=None, log_freq: int = 100):

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda" if cuda_condition else "cpu")

        self.sbert = sbert
        self.model = SBERTClassification(sbert, num_classes).to(self.device)
        self.num_classes = num_classes

        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUs for model fine-tuning" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

        self.log_freq = log_freq

    def train(self, epoch):
        train_loss = 0.0
        counter = 0
        total_correct = 0
        total_element = 0
        matrix = np.zeros([self.num_classes, self.num_classes])
        for data in self.train_dataloader:
            data = {key: value.to(self.device) for key, value in data.items()}

            classification = self.model(data["bert_input"].float(),
                                        data["time"].long(),
                                        data["bert_mask"].long())

            loss = self.criterion(classification, data["class_label"].squeeze().long())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            train_loss += loss.item()

            classification_result = classification.argmax(dim=-1)
            classification_target = data["class_label"].squeeze()
            correct = classification_result.eq(classification_target).sum().item()

            total_correct += correct
            total_element += data["class_label"].nelement()
            for row, col in zip(classification_result, classification_target):
                matrix[row, col] += 1

            counter += 1

        train_loss /= counter
        train_OA = total_correct / total_element * 100
        train_kappa = kappa(matrix)

        valid_loss, valid_OA, valid_kappa = self._validate()

        print("EP%d, train_OA=%.2f, train_Kappa=%.3f, validate_OA=%.2f, validate_Kappa=%.3f"
              % (epoch, train_OA, train_kappa, valid_OA, valid_kappa))

        return train_OA, train_kappa, valid_OA, valid_kappa

    def _validate(self):
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            total_correct = 0
            total_element = 0
            matrix = np.zeros([self.num_classes, self.num_classes])
            for data in self.valid_dataloader:
                data = {key: value.to(self.device) for key, value in data.items()}

                classification = self.model(data["bert_input"].float(),
                                            data["time"].long(),
                                            data["bert_mask"].long())

                loss = self.criterion(classification, data["class_label"].squeeze().long())
                valid_loss += loss.item()

                classification_result = classification.argmax(dim=-1)
                classification_target = data["class_label"].squeeze()

                correct = classification_result.eq(classification_target).sum().item()
                total_correct += correct
                total_element += data["class_label"].nelement()
                for row, col in zip(classification_result, classification_target):
                    matrix[row, col] += 1

                counter += 1

            valid_loss /= counter
            valid_OA = total_correct / total_element * 100
            valid_kappa = kappa(matrix)

        self.model.train()

        return valid_loss, valid_OA, valid_kappa

    def test(self, data_loader):
        with torch.no_grad():
            self.model.eval()

            total_correct = 0
            total_element = 0
            matrix = np.zeros([self.num_classes, self.num_classes])
            all_predictions = []
            all_targets = []

            for data in data_loader:
                data = {key: value.to(self.device) for key, value in data.items()}

                result = self.model(data["bert_input"].float(),
                                     data["time"].long(),
                                     data["bert_mask"].long())

                classification_result = result.argmax(dim=-1).cpu().numpy()
                classification_target = data["class_label"].squeeze().cpu().numpy()

                all_predictions.extend(classification_result)
                all_targets.extend(classification_target)

                correct = (classification_result == classification_target).sum()
                total_correct += correct
                total_element += classification_target.size
                for row, col in zip(classification_result, classification_target):
                    matrix[row, col] += 1

            test_OA = total_correct * 100.0 / total_element
            test_kappa = kappa(matrix)
            test_AA = average_accuracy(matrix)

            # Calculate F1-score, precision, and recall using sklearn
            f1 = f1_score(all_targets, all_predictions, average='macro') # Use 'macro' for unweighted average
            precision = precision_score(all_targets, all_predictions, average='macro')
            recall = recall_score(all_targets, all_predictions, average='macro')

            self.model.train()

            return test_OA, test_kappa, test_AA, precision, recall, f1, matrix, all_predictions, all_targets



    def save(self, epoch, file_path):
        output_path = file_path + "checkpoint.tar"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
        }, output_path)

        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

    def load(self, file_path):
        input_path = file_path + "checkpoint.tar"

        checkpoint = torch.load(input_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        self.model.train()
        epoch = checkpoint['epoch']

        print("EP:%d Model loaded from:" % epoch, input_path)
        return input_path


    def visualize_classification_results(self, predictions, targets, matrix):
        plt.figure(figsize=(18, 6))  # Increased figure size to accommodate all plots
    
        class_labels = ['Non-Grape', 'Grape']
    
        # Confusion Matrix
        plt.subplot(1, 3, 1)  # 1 row, 3 columns, first plot
        sns.heatmap(matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
    
        # Predictions vs True Labels
        plt.subplot(1, 3, 2)  # 1 row, 3 columns, second plot
        plt.scatter(range(len(predictions)), predictions, label='Predictions', alpha=0.5)
        plt.scatter(range(len(targets)), targets, label='True Labels', alpha=0.5)
        plt.title('Predictions vs True Labels')
        plt.xlabel('Sample Index')
        plt.ylabel('Class Label')
        plt.legend()  # Add legend
    
        # Class-wise Precision and Recall (Bar Plot)
        precision = np.diag(matrix) / np.sum(matrix, axis=0)
        recall = np.diag(matrix) / np.sum(matrix, axis=1)
        precision = np.nan_to_num(precision, nan=0.0)
        recall = np.nan_to_num(recall, nan=0.0)
    
        plt.subplot(1, 3, 3)  # 1 row, 3 columns, third plot
        width = 0.35  # Width of the bars
        x = np.arange(matrix.shape[0])  # X-axis positions
    
        plt.bar(x - width/2, precision, width, label='Precision')  # Shifted bars for better visualization
        plt.bar(x + width/2, recall, width, label='Recall')
    
        plt.xticks(x, class_labels)  # Set x-axis ticks and labels
        plt.title('Class-wise Precision and Recall')
        plt.xlabel('Class Label')
        plt.ylabel('Score')
        plt.legend()
    
        plt.tight_layout()  # Adjust layout to prevent overlapping
        plt.savefig("classification_results_names.png")
        plt.close()