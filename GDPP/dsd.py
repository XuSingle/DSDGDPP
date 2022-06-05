import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn


class DSDTraining(nn.Module):

    def __init__(self, model, sparsity, model_type, train_on_sparse=False):
        super(DSDTraining, self).__init__()

        self.model = model
        self.sparsity = sparsity
        self.train_on_sparse = train_on_sparse
        self.model_type = model_type

        # Get only conv/fc layers.
        tmp = list(self.model.named_parameters())
        # print(tmp)
        # print(tmp[0])
        self.layers = []
        if self.model_type == 'D':

            for i in range(2, len(tmp) - 1, 2):
                # print(i)
                w, b = tmp[i], tmp[i + 1]
                # print(w)
                # print(b)
                # print('---D')
                # if ("conv" in w[0] or "conv" in b[0]) or ("fc" in w[0] or "fc" in b[0]):
                self.layers.append((w[1], b[1]))
        else:
            for i in range(2, len(tmp) - 2, 2):
                # print(i)
                w, b = tmp[i], tmp[i + 1]
                # print('---')
                # print(w)
                # print(b)
                # print('---G')
                # if ("conv" in w[0] or "conv" in b[0]) or ("fc" in w[0] or "fc" in b[0]):
                self.layers.append((w[1], b[1]))

        # print(self.layers)
        # Init masks
        self.reset_masks()

    def reset_masks(self):

        self.masks = []
        for w, b in self.layers:
            mask_w = torch.ones_like(w, dtype=bool)
            mask_b = torch.ones_like(b, dtype=bool)
            self.masks.append((mask_w, mask_b))

        return self.masks

    def update_masks(self):

        for i, (w, b) in enumerate(self.layers):
            q_w = torch.quantile(torch.abs(w), q=self.sparsity)
            mask_w = torch.where(torch.abs(w) < q_w, True, False)

            q_b = torch.quantile(torch.abs(b), q=self.sparsity)
            mask_b = torch.where(torch.abs(b) < q_b, True, False)

            self.masks[i] = (mask_w, mask_b)

    def forward(self, x):
        return self.model(x)
