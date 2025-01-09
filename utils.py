import torch
import numpy as np

# Calculates the Dice coefficient for a batch of predicted and ground truth masks.
def dice_coeff_batch(batch_bn_mask, batch_true_bn_mask, device='cuda'):
    """
    Computes the Dice coefficient for a batch of binary masks.

    Args:
        batch_bn_mask (torch.Tensor): Batch of predicted binary masks.
        batch_true_bn_mask (torch.Tensor): Batch of ground truth binary masks.
        device (str): Device to perform calculations (default is 'cuda').

    Returns:
        tuple: Mean Dice coefficient and 1 - Dice coefficient.
    """
    def single_dice_coeff(input_bn_mask, true_bn_mask):
        eps = 1.0  # Small value for numerical stability
        inter_mask = torch.dot(input_bn_mask.view(-1), true_bn_mask.view(-1))
        union_mask = torch.sum(input_bn_mask.view(-1)) + torch.sum(true_bn_mask.view(-1)) + eps
        return (2 * inter_mask.float() + eps) / union_mask.float()

    dice_score = torch.FloatTensor(1).cuda(device=device).zero_() if batch_bn_mask.is_cuda else torch.FloatTensor(1).zero_()

    for pair_idx, inputs in enumerate(zip(batch_bn_mask, batch_true_bn_mask)):
        dice_score += single_dice_coeff(inputs[0], inputs[1])

    dice_batch = dice_score / (pair_idx + 1)
    return dice_batch, 1 - dice_batch

# Calculates the Tversky coefficient for a batch of predicted and ground truth masks.
def tversky_coeff_batch(batch_bn_mask, batch_true_bn_mask, alpha=0.7, beta=0.3, device='cuda'):
    """
    Computes the Tversky coefficient for a batch of binary masks.

    Args:
        batch_bn_mask (torch.Tensor): Batch of predicted binary masks.
        batch_true_bn_mask (torch.Tensor): Batch of ground truth binary masks.
        alpha (float): Weight for false negatives.
        beta (float): Weight for false positives.
        device (str): Device to perform calculations (default is 'cuda').

    Returns:
        tuple: Mean Tversky coefficient and 1 - Tversky coefficient.
    """
    def single_tversky_coeff(input_bn_mask, true_bn_mask, alpha, beta):
        eps = 0.0001  # Small value for numerical stability
        inter_mask = torch.dot(input_bn_mask.view(-1), true_bn_mask.view(-1))
        fp = torch.dot(input_bn_mask.view(-1), (1 - true_bn_mask.view(-1)))
        fn = torch.dot((1 - input_bn_mask.view(-1)), true_bn_mask.view(-1))
        union_mask = inter_mask + beta * fp + alpha * fn + eps
        return (inter_mask.float() + eps) / union_mask.float()

    tversky_score = torch.FloatTensor(1).cuda(device=device).zero_() if batch_bn_mask.is_cuda else torch.FloatTensor(1).zero_()

    for pair_idx, inputs in enumerate(zip(batch_bn_mask, batch_true_bn_mask)):
        tversky_score += single_tversky_coeff(inputs[0], inputs[1], alpha, beta)

    tversky_batch = tversky_score / (pair_idx + 1)
    return tversky_batch, 1 - tversky_batch

# Computes precision, recall, accuracy, and F1 score based on confusion matrix values.
def metrics(p_n, tp, fp, tn, fn):
    """
    Computes basic classification metrics.

    Args:
        p_n (float): Total number of samples.
        tp (float): True positives.
        fp (float): False positives.
        tn (float): True negatives.
        fn (float): False negatives.

    Returns:
        tuple: Precision, recall, accuracy, and F1 score.
    """
    try:
        accuracy = (tp + tn) / p_n
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * tp / (2 * tp + fp + fn)
    except ZeroDivisionError:
        precision, recall, accuracy, f1 = 0, 0, 0, 0
    return precision, recall, accuracy, f1

# Generates a confusion matrix and computes related metrics.
def confusion_matrix(prediction, truth):
    """
    Computes the confusion matrix and derived metrics.

    Args:
        prediction (torch.Tensor): Predicted labels.
        truth (torch.Tensor): Ground truth labels.

    Returns:
        tuple: Confusion matrix components (tp, fp, tn, fn) and classification metrics.
    """
    confusion_vector = prediction / truth
    tp = torch.sum(confusion_vector == 1).item()
    fp = torch.sum(confusion_vector == float('inf')).item()
    tn = torch.sum(torch.isnan(confusion_vector)).item()
    fn = torch.sum(confusion_vector == 0).item()
    p_n = tp + fp + tn + fn
    precision, recall, accuracy, f1 = metrics(p_n, tp, fp, tn, fn)
    return tp / p_n, fp / p_n, tn / p_n, fn / p_n, precision, recall, accuracy, f1

# Implements early stopping to terminate training if validation loss stops improving.
class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves the model when validation loss improves."""
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# Computes the positive weight for unbalanced datasets.
def pos_weight_batch(mask):
    """
    Computes the positive class weight for binary cross-entropy.

    Args:
        mask (torch.Tensor): Ground truth binary mask.

    Returns:
        float: Ratio of negative to positive samples.
    """
    size = mask.size()
    pos = torch.sum(mask)
    total_px = (size[-1]**2) * size[0]
    return (total_px - pos) / pos
