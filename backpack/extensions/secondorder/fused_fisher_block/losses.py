from functools import partial

from backpack.core.derivatives.crossentropyloss import CrossEntropyLossDerivatives
from backpack.extensions.secondorder.fused_fisher_block.fused_fisher_block_base import FusedFisherBlockBaseModule


class FusedFisherBlockLoss(FusedFisherBlockBaseModule):
    def backpropagate(self, ext, module, grad_inp, grad_out, backproped):
        # backprop symmetric factorization of the hessian of loss w.r.t. outputs of the network,
        # i.e. H = SS^T
        hess_func = self.make_loss_hessian_func(ext)

        return hess_func(module, grad_inp, grad_out)

    def make_loss_hessian_func(self, ext):
        # TODO(bmu): try both exact and MC sampling
        return self.derivatives.sqrt_hessian


class FusedFisherBlockCrossEntropyLoss(FusedFisherBlockLoss):
    def __init__(self):
        super().__init__(derivatives=CrossEntropyLossDerivatives())
