import torch

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.extensions.firstorder.fisher_block_eff.fisher_block_eff_base import FisherBlockEffBase

class FisherBlockEffLayerNorm(FisherBlockEffBase):
    def __init__(self, damping=1.0):
        self.damping = damping
        super().__init__(derivatives=BaseParameterDerivatives(), params=["bias", "weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
    	I = module.input0
    	assert(len(I.shape) in [2, 4]) # linear or conv
    	n, c, h, w = I.shape[0], I.shape[1], 1, 1 # input shape = output shape for LayerNorm
    	g_out_sc = n * g_out[0]

    	if len(I.shape) == 4: # conv
	    	h, w = I.shape[2], I.shape[3]
	    	# flatten: [n, c, h * w]
    		I = I.reshape(n, c, -1)
    		g_out_sc = g_out_sc.reshape(n, c, -1)

    	G = g_out_sc

    	grad = module.weight.grad.reshape(-1)

    	mean = I.mean(dim=-1).unsqueeze(-1)
    	var = I.var(dim=-1, unbiased=False).unsqueeze(-1)

    	x_hat = (I - mean) / (var + module.eps).sqrt()

    	J = g_out_sc * x_hat
    	J = J.reshape(J.shape[0], -1)
    	JJT = torch.matmul(J, J.t())

    	grad_prod =	torch.matmul(J, grad)

    	NGD_kernel = JJT / n
    	NGD_inv = torch.linalg.inv(NGD_kernel + self.damping * torch.eye(n).to(grad.device))
    	v = torch.matmul(NGD_inv, grad_prod)

    	gv = torch.matmul(J.t(), v) / n

    	update = (grad - gv) / self.damping
    	update = update.reshape(module.weight.grad.shape)

    	module.I = I
    	module.G = G
    	module.NGD_inv = NGD_inv

    	return update

    def bias(self, ext, module, g_inp, g_out, backproped):
    	return module.bias.grad
