from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.fused_fisher_block.fused_fisher_block_base import FusedFisherBlockBaseModule


class FusedFisherBlockLinear(FusedFisherBlockBaseModule):
    def __init__(self, damping=1.0):
        self.damping = damping
        self.alpha = alpha
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        # TODO(bmu): manully backprop quantities in the extra b/w pass

        I = module.input0
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]
        G = g_out_sc
        grad = module.weight.grad

        B =  einsum("ni,li->nl", (I, I))
        A =  einsum("no,lo->nl", (G, G))

        # compute vector jacobian product in optimization method
        grad_prod = einsum("ni,oi->no", (I, grad))
        grad_prod = einsum("no,no->n", (grad_prod, G))
        # grad_prod = 0
        out = A * B
        # out = 0
        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()

        gv = einsum("n,no->no", (v, G))
        gv = einsum("no,ni->oi", (gv, I))
        gv = gv / n

        update = (grad - gv)/self.damping

        module.I = I
        module.G = G
        module.NGD_inv = NGD_inv
        return update


    def bias(self, ext, module, g_inp, g_out, backproped):
        # TODO(bmu): manully backprop quantities in the extra b/w pass

        grad = module.bias.grad
        n = g_out[0].shape[0]
        g_out_sc = n * g_out[0]

        # compute vector jacobian product in optimization method
        grad_prod = einsum("no,o->n", (g_out_sc, grad))
        # grad_prod = 0
        out = einsum("no,lo->nl", g_out_sc, g_out_sc)
        # out = 0

        NGD_kernel = out / n
        NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
        v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
        gv = einsum("n,no->o", (v, g_out_sc))
        gv = gv / n

        update = (grad - gv)/self.damping

        return update

