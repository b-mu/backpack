from backpack.core.derivatives.batchnorm2d import BatchNorm2dDerivatives
from backpack.extensions.firstorder.fisher_block_eff.fisher_block_eff_base import FisherBlockEffBase

from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

import torch

class FisherBlockEffBatchNorm2d(FisherBlockEffBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=BatchNorm2dDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        update = torch.empty_like(module.weight.grad).copy_(module.weight.grad)
        return update
        

    def bias(self, ext, module, g_inp, g_out, backproped):
        update = torch.empty_like(module.bias.grad).copy_(module.bias.grad)
        return update
        

