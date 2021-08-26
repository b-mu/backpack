from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.fused_fisher_block.fused_fisher_block_base import FusedFisherBlockBaseModule


class FusedFisherBlockLinear(FusedFisherBlockBaseModule):
    def __init__(self, damping=1.0):
        self.damping = damping
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
        """
        y = wx + b
        g_inp: tuple of [dl/db (avg) = sum of dl/dy over batch dim, dl/dx, dl/dw]
        g_out: tuple of [dl/dy (individual, divided by batch size m)]
        backproped:
            * symmetric factorization of the hessian w.r.t. output, i.e. S in H = SS^T (scaled by 1/sqrt(m))
            * S^{(i-1)} = J^TS^{(i)}
            * jacobian of loss w.r.t. bias params = transposed jacobian of output w.r.t. bias params @ S = S
        fuse by manully backproping quantities in the extra b/w pass
        """
        # derivative of loss w.r.t. bias parameters = derivatie of loss w.r.t. layer output
        g = g_inp[0]
        m = g_out[0].shape[0]

        J = backproped.squeeze()

        # compute vector jacobian product in optimization method
        Jg = einsum("mp,p->m", (J, g))
        JTJ = einsum("mp,lp->ml", J, J)
        JTJ_inv = inv(JTJ + self.damping * eye(m).to(g.device))
        v = matmul(JTJ_inv, Jg.unsqueeze(1)).squeeze()
        gv = einsum("m,mp->p", (v, J))

        update = (g - gv) / self.damping

        return update

