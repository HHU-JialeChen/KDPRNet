# -*- coding: utf-8 -*-
from __future__ import print_function

import torch
import torch.nn as nn

class SupCluLoss(nn.Module):  
    def __init__(self, temperature=0.3,base_temperature=0.07):  
        super(SupCluLoss, self).__init__()  
        self.temperature = temperature  
        self.base_temperature = base_temperature
        mask = torch.arange(5).repeat(2).cuda()
        mask = torch.eq(mask.unsqueeze(0), mask.unsqueeze(1)).float()
        self.relation_matrix = mask.unsqueeze(0).repeat(2, 1, 1) #4,15,15
        logits_mask = 1 - torch.eye(10)  # [10,10]
        self.logits_mask = logits_mask.unsqueeze(0).repeat(2, 1, 1).float().to(torch.device('cuda'))
  
    def forward(self, features):  
        batch_size, num_samples, feature_dim = features.shape  
        anchor_dot_contrast = torch.matmul(features, features.transpose(1, 2)) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=2, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # [4,70,70]
        exp_logits = torch.exp(logits) * self.logits_mask  # [4,35,35]
        row_log_sum = torch.logsumexp(exp_logits, dim=-1)  # [4,35]
        row_log_sum_expanded = row_log_sum.unsqueeze(2).expand_as(exp_logits)
        log_prob = logits - row_log_sum_expanded  # [4,35,35]
        mean_log_prob_pos = (self.relation_matrix * log_prob).sum(2) / (self.relation_matrix.sum(2) + 1e-5)
        
        # loss
        sup_loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.mean()
          
        return sup_loss
        
