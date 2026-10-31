"""Accuracy, macro-F1, and confusion matrices in the manifest's class order."""
import torch
from .modeling import logits

def evaluate(model, kind, x, y, batch_size=128):
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        predictions = []
        for i in range(0, len(y), batch_size):
            z = logits(model, x[i:i + batch_size].to(device), kind)
            classes = z.shape[1]
            predictions.append(z.argmax(-1).cpu())
        pred = torch.cat(predictions)
        labels = y.cpu().long()
        cm = torch.bincount(labels * classes + pred, minlength=classes ** 2).reshape(classes, classes)
        f1 = (2 * cm.diag().double() / (cm.sum(0) + cm.sum(1)).clamp_min(1)).mean()
        correct = int(cm.diag().sum())
    return (dict(samples=len(y),
        correct=correct,
        accuracy_percent=100 * correct / len(y),
        macro_f1=float(f1),
        confusion_matrix=cm.tolist()),
        pred.numpy())
