import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, reduction="batchmean", temperature=1):
        super(DistillationLoss, self).__init__()
        self.reduction = reduction
        self.temperature = temperature
        
    def forward(self, student, teacher):
        loss = nn.KLDivLoss(reduction=self.reduction)(torch.log_softmax(student/self.temperature, dim=1), torch.softmax(teacher/self.temperature, dim=1))*(self.temperature**2)
        return loss