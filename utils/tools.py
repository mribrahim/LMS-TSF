import numpy as np
import torch
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred




def coverage_based_adjustment(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.05, erase_tp: bool = False) -> np.ndarray:
    """
    Optimized coverage-based adjustment for anomaly predictions.

    Parameters:
    y_true (np.ndarray): Ground truth labels (1 for anomaly, 0 for normal).
    y_pred (np.ndarray): Predicted labels (1 for anomaly, 0 for normal).
    coverage (float): Minimum coverage ratio to consider a segment as an anomaly.
    erase_tp (bool): Whether to erase true positives outside the coverage-adjusted segments.

    Returns:
    np.ndarray: Adjusted predicted labels.
    """
    y_adj = y_pred.copy()
    
    # Identify start and end indices of ground truth anomaly segments
    anomaly_boundaries = np.where(np.diff(np.pad(y_true, (1, 1), constant_values=0)) != 0)[0]
    segment_starts = anomaly_boundaries[::2]
    segment_ends = anomaly_boundaries[1::2]

    # Process each anomaly segment
    for start, end in zip(segment_starts, segment_ends):
        # Determine the width of the anomaly segment
        anomaly_width = end - start
        
        # Count predicted anomalies in this segment
        pred_count = np.sum(y_pred[start:end])
        
        # Calculate coverage ratio
        coverage_ratio = pred_count / anomaly_width
        
        if coverage_ratio >= coverage:
            # Mark entire segment as a predicted anomaly
            y_adj[start:end] = 1
        elif erase_tp:
            # Erase predictions outside valid coverage
            y_adj[start:end] = 0

    return y_true, y_adj

import pandas as pd
def sliding_window_anomaly_detection(mse_list, window_size, threshold_factor=3):
    mse_series = pd.Series(mse_list)
    
    # Calculate moving average and moving standard deviation
    moving_avg = mse_series.rolling(window=window_size, min_periods=1).mean()
    moving_std = mse_series.rolling(window=window_size, min_periods=1).std()
    
    # Calculate dynamic threshold
    dynamic_threshold = moving_avg + (threshold_factor * moving_std)
    
    # Identify anomalies
    anomalies = (mse_series > dynamic_threshold).astype(int)

    # Convert to list for output
    anomalies_list = anomalies.tolist()
    
    return anomalies_list, dynamic_threshold.tolist()