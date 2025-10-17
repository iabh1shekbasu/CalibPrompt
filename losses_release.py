import torch
import torch.nn.functional as F


def cross_entropy_loss(logits, label, **kwargs):
    """Standard cross-entropy classification loss."""
    return F.cross_entropy(logits, label)


def label_smoothing_loss(logits, label, **kwargs):
    """Cross-entropy with label smoothing; smoothness via `alpha` (default 0.1)."""
    alpha = kwargs.get("alpha", 0.1)
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in the range [0, 1).")
    target = label.to(dtype=torch.long)
    pred = logits.log_softmax(dim=-1)
    num_classes = pred.shape[-1]
    with torch.no_grad():
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(alpha / (num_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - alpha)
    return torch.mean(torch.sum(-true_dist * pred, dim=-1))


def angular_separation_loss(text_features):
    """Penalizes cosine similarity so text feature vectors spread apart."""
    n = text_features.shape[0]
    if n < 2:
        return text_features.new_zeros(())
    cos_sim_matrix = torch.matmul(text_features, text_features.T)
    mask = torch.eye(n, device=text_features.device, dtype=torch.bool)
    cos_sim_matrix = cos_sim_matrix.masked_fill(mask, 0)
    neighbor_mean = cos_sim_matrix.sum(dim=1) / (n - 1)
    return neighbor_mean.mean()


def smoothed_accuracy_and_confidence_loss(logits, label, **kwargs):
    """Matches model confidence with label-smoothed class frequencies."""
    alpha = kwargs.get("alpha", 0.05)
    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in the range [0, 1).")
    output = torch.softmax(logits, dim=1)
    classes = output.shape[1]

    loss = logits.new_zeros(())
    for c in range(classes):
        avg_count = (label == c).float().mean()
        smooth_freq = avg_count * (1 - alpha) + (1 - avg_count) * (alpha / (classes - 1))
        avg_conf = output[:, c].mean()
        loss = loss + torch.abs(avg_conf - smooth_freq)

    loss = loss / classes
    return loss
