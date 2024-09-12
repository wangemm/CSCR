import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CLoss(nn.Module):
    def __init__(self, t, normalize_loss, class_num, device):
        super(CLoss, self).__init__()
        self.t = t
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.class_num = class_num
        self.normalize = normalize_loss
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.mask = self.mask_correlated_clusters(class_num)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def semi_contrast_loss(self, v1, v2, we1, we2, sim_label):

        # mask the unavailable instances
        mask_miss_inst = we1.mul(we2).bool()
        v1 = v1[mask_miss_inst]
        v2 = v2[mask_miss_inst]
        sim_label = sim_label[mask_miss_inst]
        sim_label = sim_label[:, mask_miss_inst]

        sim_label = sim_label - torch.diag_embed(torch.diag(sim_label))
        sim_label = sim_label + torch.eye(sim_label.size(0)).to(self.device)
        # s_n = sim_label.data.cpu().numpy()
        n = v1.size(0)
        N = 2 * n
        if n == 0:
            return 0
        # normalize two vectors
        if self.normalize:
            v1 = F.normalize(v1, p=2, dim=1)
            v2 = F.normalize(v2, p=2, dim=1)
        z = torch.cat((v1, v2), dim=0)
        similarity_mat = torch.div(torch.matmul(z, z.T), self.t)
        logits_max, _ = torch.max(similarity_mat, dim=1, keepdim=True)
        logits = similarity_mat - logits_max.detach()

        sim_label = sim_label.repeat(2, 2)
        logits_mask = torch.scatter(
            torch.ones_like(sim_label),
            1,
            torch.arange(v1.shape[0] * 2).view(-1, 1).to(self.device),
            0
        )
        sim_label = sim_label * logits_mask
        # s_n = sim_label.data.cpu().numpy()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (sim_label * log_prob).sum(1) / sim_label.sum(1)
        loss = - (self.t / self.t) * mean_log_prob_pos
        loss = loss.view(2, v1.shape[0]).mean()
        return loss

