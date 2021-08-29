from backpack.extensions.module_extension import ModuleExtension


class FusedFisherBlockBaseModule(ModuleExtension):
    def __init__(self, derivatives, params=None):
        super().__init__(params=params)
        self.derivatives = derivatives

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        H_inv, J, (m, c) = backproped
        return [H_inv, self.derivatives.jac_t_mat_prod(module, g_inp, g_out, J), (m, c)]
