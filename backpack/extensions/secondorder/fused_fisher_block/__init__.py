from torch.nn import (
    Linear
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    linear
)

class FusedFisherBlock(BackpropExtension):
    def __init__(self, loss_sample, damping=1.0):
        self.loss_sample = loss_sample
        self.damping = damping
        super().__init__(
            savefield="fused_fisher_block",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.FusedFisherBlockLinear(self.damping)
            },
        )
