from backpack.core.derivatives.relu import ReLUDerivatives
from backpack.extensions.secondorder.fused_fisher_block.fused_fisher_block_base import FusedFisherBlockBaseModule


class FusedFisherBlockReLU(FusedFisherBlockBaseModule):
    def __init__(self):
        super().__init__(derivatives=ReLUDerivatives())
