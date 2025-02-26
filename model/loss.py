import torch
import torch.nn as nn

def create_targets_stage1(future_activities, num_activities):
    """
    Args:
        future_activities: Tensor of shape (Batch, ws), values in 0 (padding) or 1 to n.
        num_activities: int, the number of unique activities (n).
    
    Returns:
        targets_stage1: Tensor of shape (Batch, num_activities), multi-label targets for stage 1.
    """
    batch_size, ws = future_activities.shape

    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, ws).reshape(-1)
    # activity_indices-1 (1-n)
    activity_indices = (future_activities - 1).reshape(-1)

    mask = activity_indices >= 0
    batch_indices = batch_indices[mask]
    activity_indices = activity_indices[mask]

    targets_stage1 = torch.zeros(batch_size, num_activities, device=future_activities.device, dtype=torch.float32)

    targets_stage1[batch_indices, activity_indices] = 1.0
    return targets_stage1

def create_targets_stage2(future_activities, num_activities):
    """
    Args:
        future_activities: Tensor of shape (Batch, ws), values in 0 (padding) or 1 to n.
        num_activities: int, the number of unique activities (n).
    
    Returns:
        targets_stage2: Tensor of shape (Batch,), the groud truth for the whole process.
    """
    batch_size, ws = future_activities.shape
    
    mask = future_activities > 0
    indices = torch.arange(ws, device=future_activities.device).unsqueeze(0).expand(batch_size, ws)  # (Batch, ws)

    indices = torch.where(mask, indices, torch.full_like(indices, num_activities+1, device=future_activities.device))
    
    min_indices, _ = indices.min(dim=1)  # (Batch,)
    # (Batch,) 0-n-1
    targets_stage2 = future_activities[torch.arange(batch_size), min_indices]-1
    assert targets_stage2.min() >= 0 and targets_stage2.max() < num_activities, "targets_stage2 超出有效范围"
    return targets_stage2

class ESFLoss(nn.Module):
    def __init__(self, alpha=0.1, exp_factor=2):
        super(ESFLoss, self).__init__()
        self.alpha = alpha          # 第一阶段损失的权重
        self.exp_factor = exp_factor
        # 定义第一阶段的损失函数（二元交叉熵损失）
        self.bce_loss = nn.BCELoss()
        # 定义第二阶段的损失函数（交叉熵损失）
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        """
        outputs: enable activity (batch_size, num_activities), prediction probility(batch_size, num_activities)
        targets: activities in future windows (batch_size, future_ws)
        """
        enable_state, prediction = outputs
        num_activities = enable_state.shape[1]
        targets_stage1 = create_targets_stage1(targets, num_activities)
        targets_stage2 = create_targets_stage2(targets, num_activities)
        # loss
        loss_stage1 = self.bce_loss(enable_state, targets_stage1)
        weighted_loss_stage1  = (loss_stage1 * torch.exp(self.exp_factor * enable_state)).mean()
        loss_stage2 = self.ce_loss(prediction, targets_stage2)
        
        # 总损失
        total_loss = self.alpha * weighted_loss_stage1  + loss_stage2
        
        return self.alpha * weighted_loss_stage1, loss_stage2, total_loss

