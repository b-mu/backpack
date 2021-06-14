from torch.nn import (
    Conv1d,
    Conv2d,
    Linear,
    BatchNorm1d,
    BatchNorm2d,
    LayerNorm
)

from backpack.extensions.backprop_extension import BackpropExtension

from . import (
    conv1d,
    conv2d,
    linear,
    batchnorm1d,
    batchnorm2d,
    layernorm
)


class FisherBlockEff(BackpropExtension):
    

    def __init__(self, damping=1.0, alpha=0.95, low_rank='false', gamma=0.95, memory_efficient='false', super_opt='false'):
        self.gamma = gamma
        self.damping = damping
        self.alpha =alpha
        self.low_rank = low_rank
        self.memory_efficient = memory_efficient
        self.super_opt = super_opt
        super().__init__(
            savefield="fisher_block",
            fail_mode="WARNING",
            module_exts={
                Linear: linear.FisherBlockEffLinear(self.damping, self.alpha),
                Conv1d: conv1d.FisherBlockEffConv1d(self.damping),
                Conv2d: conv2d.FisherBlockEffConv2d(self.damping, self.low_rank, self.gamma, self.memory_efficient, self.super_opt),
                BatchNorm1d: batchnorm1d.FisherBlockEffBatchNorm1d(self.damping),
                BatchNorm2d: batchnorm2d.FisherBlockEffBatchNorm2d(self.damping),
                LayerNorm: layernorm.FisherBlockEffLayerNorm(self.damping)
            },
        )
