from datetime import datetime
import argparse
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from preprocessing import apply_smote_to_subjects, process_subjects, load_all_subject_data
from encoding.GAFGenerator import GAFDataset
from encoding.MTFGenerator import MTFDataset
from encoding.RPGenerator import RPDataset
from model.CNN import get_model
from utils.mail import send_email
from utils.plot import plot_accuracy, plot_loss, plot_confusion_matrix

torch.backends.cudnn.benchmark = True

SUBJECTS = [
    'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17'
]

FEATURES = [
    'chest_acc_x', 'chest_acc_y', 'chest_acc_z', 'chest_ecg', 'chest_eda', 'chest_emg', 'chest_resp',
    'chest_temp', 'wrist_acc_x', 'wrist_acc_y', 'wrist_acc_z', 'wrist_bvp', 'wrist_eda', 'wrist_temp'
]


def main(frequency, dataset_path, window_sec, window_stride, patience, epochs, methods, model_choice, labels):
    dataset_folder = dataset_path
    resample_freq = frequency
    preprocess_output_folder = f'{dataset_folder}/Preprocessed_Subjects'
    model_output_folder = 'Results/'
    window_size = window_sec * resample_freq
    time_step = window_stride * resample_freq
    seed = 42
    batch_size = 32
    image_size = 224
    learning_rate = 0.0012
    optimizer_choice = 'Adam'
    n_bins = 3
    threshold = 0.5

    n_classes = len(labels) - 1
    print(f'Number of classes: {n_classes}')

    resample_needed = 'rp_dtw' not in methods
    print(f'Resampling needed: {resample_needed}')

    process_subjects(frequency, labels, window_size, preprocess_output_folder=preprocess_output_folder,
                     dataset_folder=dataset_folder, resample_needed=resample_needed)
    all_subject_data = load_all_subject_data(frequency, subjects=SUBJECTS,
                                             preprocess_output_folder=preprocess_output_folder,
                                             resample_needed=resample_needed)

    train_subjects, validation_subjects = train_test_split(SUBJECTS, test_size=3, random_state=seed)
    train_data = {subj: all_subject_data[subj] for subj in train_subjects}

    resampled_train_data = apply_smote_to_subjects(train_data)
    for subj in train_subjects:
        all_subject_data[subj] = resampled_train_data[subj]

    print(f'Train subjects: {train_subjects}')
    print(f'Validation subjects: {validation_subjects}')

    if 'gaf_difference' in methods:
        train_dataset = GAFDataset(train_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                   method='difference', window_size=window_size, time_step=time_step, seed=seed)
        val_dataset = GAFDataset(validation_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                 method='difference', window_size=window_size, time_step=time_step, seed=seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        start_training(train_loader, val_loader, model_choice, 'GAF difference', n_classes=n_classes, channels=14,
                       patience=patience, epochs=epochs, optimizer_choice=optimizer_choice, learning_rate=learning_rate,
                       model_output_folder=model_output_folder, resample_freq=resample_freq, window_sec=window_sec)

    if 'gaf_summation' in methods:
        train_dataset = GAFDataset(train_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                   method='summation', window_size=window_size, time_step=time_step, seed=seed)
        val_dataset = GAFDataset(validation_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                 method='summation', window_size=window_size, time_step=time_step, seed=seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        start_training(train_loader, val_loader, model_choice, 'GAF summation', n_classes=n_classes, channels=14,
                       patience=patience, epochs=epochs, optimizer_choice=optimizer_choice, learning_rate=learning_rate,
                       model_output_folder=model_output_folder, resample_freq=resample_freq, window_sec=window_sec)

    if 'mtf' in methods:
        train_dataset = MTFDataset(train_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                   window_size=window_size, time_step=time_step, n_bins=n_bins, seed=seed)
        val_dataset = MTFDataset(validation_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                 window_size=window_size, time_step=time_step, n_bins=n_bins, seed=seed)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        start_training(train_loader, val_loader, model_choice, 'MTF', n_classes=n_classes, channels=14,
                       patience=patience, epochs=epochs, optimizer_choice=optimizer_choice, learning_rate=learning_rate,
                       model_output_folder=model_output_folder, resample_freq=resample_freq, window_sec=window_sec)

    if 'rp_euclidean' in methods:
        train_dataset = RPDataset(train_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                  threshold=threshold, window_size=window_size, time_step=time_step, shuffle=True,
                                  seed=seed, distance='euclidean')
        val_dataset = RPDataset(validation_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                threshold=threshold, window_size=window_size, time_step=time_step, shuffle=True,
                                seed=seed, distance='euclidean')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        start_training(train_loader, val_loader, model_choice, 'RP euclidean', n_classes=n_classes, channels=1,
                       patience=patience, epochs=epochs, optimizer_choice=optimizer_choice, learning_rate=learning_rate,
                       model_output_folder=model_output_folder, resample_freq=resample_freq, window_sec=window_sec)

    if 'rp_dtw' in methods:
        train_dataset = RPDataset(train_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                  threshold=threshold, window_size=window_size, time_step=time_step, shuffle=True,
                                  seed=seed, distance='dtw')
        val_dataset = RPDataset(validation_subjects, all_subject_data, FEATURES, labels, image_size=image_size,
                                threshold=threshold, window_size=window_size, time_step=time_step, shuffle=True,
                                seed=seed, distance='dtw')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        start_training(train_loader, val_loader, model_choice, 'RP dtw', n_classes=n_classes, channels=1,
                       patience=patience, epochs=epochs, optimizer_choice=optimizer_choice, learning_rate=learning_rate,
                       model_output_folder=model_output_folder, resample_freq=resample_freq, window_sec=window_sec)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    y_true_train = []
    y_pred_train = []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device).float()

        optimizer.zero_grad()
        outputs = model(inputs).squeeze(1)  # logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        predicted = (torch.sigmoid(outputs) > 0.5).int()
        train_total += labels.size(0)
        train_correct += (predicted == labels.int()).sum().item()

        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

    train_loss /= len(train_loader.dataset)
    train_accuracy = train_correct / train_total

    return train_loss, train_accuracy, y_true_train, y_pred_train


def validate_one_epoch(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    y_true_val = []
    y_pred_val = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            predicted = (torch.sigmoid(outputs) > 0.5).int()
            val_total += labels.size(0)
            val_correct += (predicted == labels.int()).sum().item()

            y_true_val.extend(labels.cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_accuracy = val_correct / val_total

    return val_loss, val_accuracy, y_true_val, y_pred_val


def start_training(train_loader, val_loader, model_choice, name='', n_classes=1, channels=14, patience=10, epochs=100,
                   optimizer_choice='Adam', learning_rate=0.0012, model_output_folder='results', resample_freq=30,
                   window_sec=60):
    device = get_device()
    print(f'DEVICE {device}')
    start_time = time.time()

    logs = f'logs/{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir=logs)

    model, optimizer, criterion = get_model(model_choice, optimizer_choice, learning_rate, n_classes, channels)
    model = model.to(device)

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    y_true = {'train': [], 'val': []}
    y_pred = {'train': [], 'val': []}

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_accuracy, y_true_train, y_pred_train = train_one_epoch(model, train_loader, criterion,
                                                                                 optimizer, device)
        val_loss, val_accuracy, y_true_val, y_pred_val = validate_one_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)

        y_true['train'].extend(y_true_train)
        y_pred['train'].extend(y_pred_train)
        y_true['val'].extend(y_true_val)
        y_pred['val'].extend(y_pred_val)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_accuracy, epoch)

        print(
            f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_output_folder}/{name}_best_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print('Early stopping triggered')
            break

    writer.close()
    path = f'{model_output_folder}/{resample_freq}/{name}_{epochs}_{window_sec}_model'
    total_time = time.time() - start_time
    print(f'Training completed in {total_time // 60}m {total_time % 60}s')

    log_string = (
        f'Model: {name}\n'
        f'Args: FREQ {resample_freq}\nEPOCHS: {epochs}\nWINDOW_SEC: {window_sec}\n'
        f'{device}\n'
        f'Best Accuracy Train: {min(history["train_acc"]):.4f}\n'
        f'Best Accuracy Validation: {min(history["val_acc"]):.4f}\n'
        f'Best Loss Train: {min(history["train_loss"]):.4f}\n'
        f'Best Loss Validation: {min(history["val_acc"]):.4f}\n'
        f'All Train Accuracies: {history["train_acc"]}\n'
        f'All Validation Accuracies: {history["val_acc"]}\n'
        f'All Train Losses: {history["train_loss"]}\n'
        f'All Validation Losses: {history["val_loss"]}\n'
    )

    loss_path = plot_loss(history['train_loss'], history['val_loss'], name, path)
    acc_path = plot_accuracy(history['train_acc'], history['val_acc'], name, path)
    cm_path = plot_confusion_matrix(y_true['val'], y_pred['val'], name, path)
    send_email(f'Training {name} completed', f'Log: {log_string}', acc_path, loss_path, cm_path)
    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freq', type=int, default=700, help='Frequency of the preprocessing')
    parser.add_argument('--dataset', type=str, default='data/WESAD', help='Path to the dataset')
    parser.add_argument('--sec', type=int, default=60, help='Window in seconds')
    parser.add_argument('--window_stride', type=int, default=1, help='Window stride in seconds')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['gaf_difference', 'gaf_summation', 'mtf', 'rp_euclidean', 'rp_dtw'],
                        help='Methods to run')
    parser.add_argument('--model', type=str, choices=['custom', 'vgg', 'resnet'], default='custom',
                        help='Model choice for training')
    parser.add_argument('--labels', type=int, nargs='+', default=[1, 2], help='List of labels for classification')
    args = parser.parse_args()

    if args.freq and args.dataset and args.sec:
        main(args.freq, args.dataset, args.sec, args.window_stride, args.patience, args.epochs, args.methods,
             args.model, args.labels)
    else:
        print('Please provide the frequency and the path to the dataset')
