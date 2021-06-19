from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.firstorder.fisher_block_eff.fisher_block_eff_base import FisherBlockEffBase
from torch import einsum, matmul, sum, numel, sqrt, norm, eye, randint, cumsum, diag
from torch.nn import Unfold, MaxPool2d, AvgPool2d
from torch.nn.functional import conv1d, conv2d, conv3d
from backpack.utils.ein import eingroup
from backpack.utils.conv import unfold_func
from torch.cuda import empty_cache
from torch.linalg import inv, svd


# import numpy as np
# import seaborn as sns
# import matplotlib.pylab as plt
MODE = 0
class FisherBlockEffConv2d(FisherBlockEffBase):
    def __init__(self, damping=1.0, low_rank='false', gamma=0.95, memory_efficient='false', super_opt='false', save_kernel='false'):
        self.damping = damping
        self.low_rank = low_rank
        self.gamma = gamma
        self.memory_efficient = memory_efficient
        self.super_opt = super_opt
        self.save_kernel = save_kernel
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, bpQuantities):
        if MODE == 0: # my implementation

            grad = module.weight.grad
            # print(grad.shape)
            grad_reshape = grad.reshape(grad.shape[0], -1)
            n = g_out[0].shape[0]
            g_out_sc = n * g_out[0]

            I = unfold_func(module)(module.input0)
            grad_output_viewed = g_out_sc.reshape(g_out_sc.shape[0], g_out_sc.shape[1], -1)
            G = grad_output_viewed

            N = I.shape[0]
            K = I.shape[1]
            L = I.shape[2]
            M = G.shape[1]
            # print(N,K,L,M)
            flag = False
            if self.super_opt == 'true':
                flag = N * (L * L) * (K + M) < K * M * L + N * K * M
            else:
                flag = (L * L) * (K + M) < K * M

            if flag == True :
                II = einsum("nkl,qkp->nqlp", (I, I))
                GG = einsum("nml,qmp->nqlp", (G, G))
                out = einsum('nqlp->nq', II * GG) 
                x1 = einsum("nkl,mk->nml", (I, grad_reshape))
                grad_prod = einsum("nml,nml->n", (x1, G)) 
                NGD_kernel = out / n
                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                gv = einsum("n,nml->nml", (v, G))
                gv = einsum("nml,nkl->mk", (gv, I))
                gv = gv.view_as(grad)
                gv = gv / n

                module.NGD_inv = NGD_inv
                if self.memory_efficient == 'true':
                    module.I = module.input0
                else:
                    module.I = I
                module.G = G
                del I
                del G
                del II
                del GG
                empty_cache()
                
            else:
                AX = einsum("nkl,nml->nkm", (I, G))
                del I
                del G
                AX_ = AX.reshape(n , -1)
                NGD_kernel = matmul(AX_, AX_.t()) / n 
                ### testing low-rank
                if self.low_rank == 'true':
                    V, S, U = svd(AX_.T, compute_uv=True, full_matrices=False)
                    U = U.t()
                    V = V.t()
                    
                    cs = cumsum(S, dim = 0)
                    sum_s = sum(S)
                    index = ((cs - self.gamma * sum_s) <= 0).sum()
                    U = U[:, 0:index]
                    S = S[0:index]
                    V = V[0:index, :]
                    
                    module.U = U
                    module.S = S
                    module.V = V
                del AX_

                grad_prod = einsum("nkm,mk->n", (AX, grad_reshape))

                NGD_inv = inv(NGD_kernel + self.damping * eye(n).to(grad.device))
                module.NGD_inv = NGD_inv 
                v = matmul(NGD_inv, grad_prod.unsqueeze(1)).squeeze()
                del NGD_inv
                empty_cache()
                gv = einsum("nkm,n->mk", (AX, v)).view_as(grad) /n
                module.AX = AX

            if self.save_kernel == 'true':
                module.NGD_kernel = NGD_kernel
                
            update = (grad - gv)/self.damping
            return update
        

    
    def bias(self, ext, module, g_inp, g_out, bpQuantities):
        
        return module.bias.grad
        



