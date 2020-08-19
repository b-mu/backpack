from test.extensions.implementation.base import ExtensionsImplementation

import backpack.extensions as new_ext
from backpack import backpack


class BackpackExtensions(ExtensionsImplementation):
    """Extension implementations with BackPACK."""

    def __init__(self, problem):
        problem.extend()
        super().__init__(problem)

    def batch_grad(self):
        with backpack(new_ext.BatchGrad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_grads = [p.grad_batch for p in self.problem.model.parameters()]
        return batch_grads

    def batch_dot_grad(self):
        with backpack(new_ext.BatchDotGrad()):
            _, _, loss = self.problem.forward_pass()
            loss.backward()
            batch_dots = [p.batch_dot for p in self.problem.model.parameters()]
        return batch_dots
