from torch import einsum, eye, matmul, ones_like, norm
from torch.linalg import inv

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.fused_fisher_block.fused_fisher_block_base import FusedFisherBlockBaseModule


class FusedFisherBlockLinear(FusedFisherBlockBaseModule):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        """
        y = wx + b
        g_inp: tuple of [dl/db (avg) = sum of dl/dy over batch dim, dl/dx, dl/dw]
        g_out: tuple of [dl/dy (individual, divided by batch size m)]
        backproped B:
            * [c(number of classes), m(batch size), o(number of outputs)]
            * batched symmetric factorization of G(y) = J^T H J (scaled by 1/sqrt(m), where
                * J is the Jacobian of network outputs w.r.t. y
                * H is the Hessian of loss w.r.t network outputs
                * so initially the symmetric factorization of Hessian of loss w.r.t. network outputs, i.e. S in H = SS^T
                * backpropagation to the previous layer by left multiplying the Jacobian of y w.r.t. x
            * batched symmetric factorization of GGN/FIM G(w) = transposed Jacobian of output y w.r.t. weight params w @ B
        fuse by manully backproping quantities in the extra b/w pass
        """
        m = g_out[0].shape[0]

        I = module.input0
        G = backproped.squeeze() # mc_samples = 1, scaled by 1/sqrt(m)
        g = g_inp[2] # g = dw = einsum("mo,mi->io", (g_out[0], I))

        # compute the covariance factors II and GG
        II =  einsum("mi,li->ml", (I, I))
        GG =  einsum("mo,lo->ml", (G, G))

        # ngd update + SMW formula = J^T @ inv(JJ^T + damping * I) @ Jg
        Jg = einsum("mi,io->mo", (I, g))
        Jg = einsum("mo,mo->m", (Jg, G))
        JJT = II * GG
        JJT_inv = inv(JJT + self.damping * eye(m).to(g.device))
        v = matmul(JJT_inv, Jg.unsqueeze(1)).squeeze()
        gv = einsum("m,mo->mo", (v, G))
        gv = einsum("mo,mi->oi", (gv, I))

        update = (g.t() - gv) / self.damping

        module.I = I
        module.G = G
        module.NGD_inv = JJT_inv

        return update


    def bias(self, ext, module, g_inp, g_out, backproped):
        """
        y = wx + b
        g_inp: tuple of [dl/db (avg) = sum of dl/dy over batch dim, dl/dx, dl/dw]
        g_out: tuple of [dl/dy (individual, divided by batch size m)]
        backproped B:
            * [c(number of classes), m(batch size), o(number of outputs)]
            * batched symmetric factorization of G(y) = J^T H J (scaled by 1/sqrt(m), where
                * J is the Jacobian of network outputs w.r.t. y
                * H is the Hessian of loss w.r.t network outputs
                * so initially the symmetric factorization of Hessian of loss w.r.t. network outputs, i.e. S in H = SS^T
                * backpropagation to the previous layer by left multiplying the Jacobian of y w.r.t. x
            * batched symmetric factorization of GGN/FIM G(b) = transposed Jacobian of output y w.r.t. bias params b @ B = B
        fuse by manully backproping quantities in the extra b/w pass
        """
        g = g_inp[0]
        m = g_out[0].shape[0]

        J = backproped.squeeze() # mc_samples = 1

        # ngd update + SMW formula = J^T @ inv(JJ^T + damping * I) @ Jg
        Jg = einsum("mp,p->m", (J, g))
        JJT = einsum("mp,lp->ml", J, J)
        JJT_inv = inv(JJT + self.damping * eye(m).to(g.device))
        v = matmul(JJT_inv, Jg.unsqueeze(1)).squeeze()
        gv = einsum("m,mp->p", (v, J))

        update = (g - gv) / self.damping

        return update

