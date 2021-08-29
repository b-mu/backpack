from torch.nn import (
    CrossEntropyLoss,
    Linear,
    ReLU,
    Flatten
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    linear,
    losses,
    activations,
    flatten
)

class FusedFisherBlock(BackpropExtension):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(
            savefield="fused_fisher_block",
            fail_mode="WARNING",
            module_exts={
                CrossEntropyLoss: losses.FusedFisherBlockCrossEntropyLoss(),
                Linear: linear.FusedFisherBlockLinear(self.damping),
                ReLU: activations.FusedFisherBlockReLU(),
                Flatten: flatten.FusedFisherBlockFlatten()
            },
        )
