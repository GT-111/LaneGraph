import torch.nn.functional as F
import torch

def focal_loss(pred, gt, alpha=2, beta=4):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    pos_loss = -((1 - pred) ** alpha) * torch.log(pred + 1e-9) * pos_inds
    neg_loss = -((1 - gt) ** beta) * (pred ** alpha) * torch.log(1 - pred + 1e-9) * neg_inds

    return (pos_loss + neg_loss).mean()

def dice_loss(pred, target):
    pred = pred.sigmoid()
    smooth = 1.
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def cosine_similarity_loss(pred, gt):
    pred_norm = F.normalize(pred, dim=1)
    gt_norm = F.normalize(gt, dim=1)
    return 1 - (pred_norm * gt_norm).sum(dim=1).mean()

def l1_loss(pred, gt, mask=None):
    if mask is not None:
        return F.l1_loss(pred * mask, gt * mask)
    return F.l1_loss(pred, gt)



def compute_total_loss(pred, target, weights=None):
    """Compute total loss with configurable weights for each component.
    
    Args:
        pred: Dictionary of model predictions
        target: Dictionary of ground truth targets
        weights: Optional dictionary of loss weights. If None, uses default weights.
    """
    if weights is None:
        weights = {
            "heatmap": 1.0,
            "keypoint_cls": 0.5, 
            "edge_vec": 1.0,
            "edge_prob": 1.0,
            "edge_cls": 0.5,
            "segmentation": 0.5,
            "topo_tensor": 1.0,
            "topology_softIOU": 0.1
        }

    losses = {}

    # Node/keypoint losses
    losses["L_heatmap"] = weights["heatmap"] * focal_loss(
        pred["node_heatmap"], target["node_heatmap"]
    )
    losses["L_keypoint_cls"] = weights["keypoint_cls"] * F.cross_entropy(
        pred["node_type"], target["node_type"]
    )

    # Edge losses  
    losses["L_edge_vec"] = weights["edge_vec"] * cosine_similarity_loss(
        pred["edge_vector"], target["edge_vector"]
    )
    losses["L_edge_prob"] = weights["edge_prob"] * F.binary_cross_entropy(
        pred["edge_prob"], target["edge_prob"]
    )
    losses["L_edge_cls"] = weights["edge_cls"] * F.cross_entropy(
        pred["edge_class"], target["edge_class"]
    )

    # Segmentation loss
    losses["L_segmentation"] = weights["segmentation"] * dice_loss(
        pred["segmentation"], target["segmentation"]
    )

    # Topological losses
    losses["L_topo_tensor"] = weights["topo_tensor"] * l1_loss(
        pred["topo_tensor"], target["topo_tensor"]
    )

    # Optional topology regularization
    if "L_topology_softIOU" in target:
        losses["L_topology_softIOU"] = weights["topology_softIOU"] * target["L_topology_softIOU"]

    total = sum(losses.values())
    return total, losses