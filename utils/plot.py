import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def plot_loss(train_losses, val_losses, name, path):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Model Loss {name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend(loc='upper left')
    plt.savefig(f'{path}/{name}_loss.png')
    plt.close()
    return f'{path}/{name}_loss.png'


def plot_accuracy(train_accuracies, val_accuracies, name, path):
    plt.figure(figsize=(12, 6))
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Model Accuracy {name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.ylim(0, 1)
    plt.legend(loc='upper left')
    plt.savefig(f'{path}/{name}_accuracy.png')
    plt.close()
    return f'{path}/{name}_accuracy.png'


def plot_confusion_matrix(y_true, y_pred, technique_name, output_folder):
    # Label mapping
    label_mapping = {'Baseline': 0, 'Stress': 1, 'Amusement': 2, 'Meditation': 3}

    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    labels_text = [label for label, index in label_mapping.items() if index in unique_labels]
    label_indices = [index for index in label_mapping.values() if index in unique_labels]

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=label_indices)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_text)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix for {technique_name}')
    plt.savefig(f'{output_folder}/{technique_name}_confusion_matrix.png')
    plt.close()

    return f'{output_folder}/{technique_name}_confusion_matrix.png'
