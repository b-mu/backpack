from torch import einsum
# TODO: remove
from torch import rand
import opt_einsum as oe
from torch import zeros
from torch.nn.functional import dropout

import torch
import numpy as np
def extract_weight_diagonal(module, backproped):
    return einsum("vno,ni->oi", (backproped ** 2, module.input0 ** 2))


def extract_bias_diagonal(module, backproped):
    return einsum("vno->o", backproped ** 2)

# TODO: Add support for NGD here
def extract_weight_ngd(module, backproped, MODE):
	#### exact methods ####
	# test: naive method plus
    # A =  einsum("vno,ni->vnoi", (backproped, module.input0))
    # return  einsum("vnoi,kloi->vnkl", (A, A))

    # test: me plus [GOLD]
    if MODE == -1: # silent mode to avoid doing any extra work here, only return 0
        v = backproped.shape[0]
        n = backproped.shape[1]
        return zeros(v*n,v*n).to(module.input0.device)
    elif MODE == 7: # test the order
        B =  einsum("ni,li->nl", (module.input0, module.input0))   
        # print('B', B) 
        A =  einsum("vno,klo->vnkl", (backproped, backproped))
        # print('A', A)
        return einsum("vnkl,nl->vnkl", (A, B))
    elif MODE == 17: # add dropout in backward pass for large linear layers
        # this is a sampling technique
        inp = module.input0
        l = inp.shape[1]
        prob = 0.1
        l_new = int(np.floor(prob * l))

        # print('input to linear layer before droput:', inp.shape)
        Borg = einsum("ni,li->nl", (inp, inp)) 

        if inp.shape[1] > 7000:
            inp =  inp[:, torch.randint(l, (l_new,))] 

        B =  einsum("ni,li->nl", (inp, inp)) / ( prob)
        # print(torch.norm(B - Borg)/torch.norm(Borg))

        A =  einsum("vno,klo->vnkl", (backproped, backproped))
        return einsum("vnkl,nl->vnkl", (A, B))
    elif MODE == 13: # testing block diagonal version
        B =  einsum("ni,li->nl", (module.input0, module.input0))    
        A =  einsum("vno,vlo->vnl", (backproped, backproped))
        return einsum("vnl,nl->vnl", (A, B))
    else:
        B =  einsum("ni,li->nl", (module.input0, module.input0))	
        A =  einsum("vno,klo->vnkl", (backproped, backproped))
        return einsum("vnkl,nl->vnkl", (A, B))

    # test: me plus plus [SILVER]
    # A = einsum("ni,li,vno,klo->vnkl", (module.input0, module.input0, backproped, backproped))
    # return A

    # test: opt_einsum
    # A = oe.contract("ni,li,vno,klo->vnkl", module.input0, module.input0, backproped, backproped)
    # return A

    #### extra approximations ####
    # test: only diagonals:
    # A = einsum("vno,ni->vnoi", (backproped ** 2, module.input0 ** 2))
    # return einsum("vnoi->vn", A)

def extract_bias_ngd(module, backproped, MODE):
    if MODE == -1: # silent mode, only backpropagating Jacobians
        v = backproped.shape[0]
        n = backproped.shape[1]
        return zeros(v*n,v*n).to(module.input0.device)
    elif MODE == 7 or MODE == 17: # test the order
        return einsum("vno,klo->vnkl", backproped, backproped)
    elif MODE == 13: # test the block version
        return einsum("vno,vlo->vnl", backproped, backproped)
    else: # normal mode
        return einsum("vno,klo->vnkl", backproped, backproped)
    # return einsum("vno->vn", backproped ** 2)

