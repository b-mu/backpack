from functools import partial

from torch.linalg import inv
from torch import einsum, eye

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.extensions.secondorder.fused_fisher_block.fused_fisher_block_base import FusedFisherBlockBaseModule


class FusedFisherBlockLoss(FusedFisherBlockBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        # backprop symmetric factorization of the hessian of loss w.r.t. the network outputs,
        # i.e. S in H = SS^T
        hess_func = self.make_loss_hessian_func(ext)
        sqrt_H = hess_func(module, grad_inp, grad_out)
        c_, m, c = sqrt_H.size()
        H = einsum('omc,olv->cmvl', (sqrt_H, sqrt_H)).reshape(c * m, c * m)
        H_inv = inv(H)

        return (H_inv, eye(c, c * m).to(H_inv.device), (m, c))

    def make_loss_hessian_func(self, ext):
        # TODO(bmu): try both exact and MC sampling
        # set mc_samples = 1 for backprop efficiency
        return self.derivatives.sqrt_hessian


class FusedFisherBlockCrossEntropyLoss(FusedFisherBlockLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
