"""Training of c2d2 on MNIST with SGD and CGN."""

import numpy
import torch
from torch.nn import CrossEntropyLoss, ReLU, Sigmoid, Tanh
from os import path
from collections import OrderedDict
from exp.loading.load_mnist import MNISTLoader
from exp.training.second_order import SecondOrderTraining
from bpexts.optim.cg_newton import CGNewton
from bpexts.cvp.sequential import convert_torch_to_cvp
from exp.training.runner import TrainingRunner
from exp.utils import (directory_in_data, dirname_from_params)
from exp.models.convolution import mnist_c2d2

# directories
dirname = 'exp07_c2d2_optimization/cvp'
data_dir = directory_in_data(dirname)

# global hyperparameters
epochs = 5
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logs_per_epoch = 10
test_batch = 500

# mapping from strings to activation functions
activation_dict = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}


def mnist_cgnewton_train_fn(batch, modify_2nd_order_terms, activation, lr,
                            alpha, cg_maxiter, cg_tol, cg_atol):
    """Create training instance for MNIST CG experiment.


    Parameters:
    -----------
    batch : int
        Batch size
    modify_2nd_order_terms : str
        Strategy for treating 2nd-order effects of module functions:
        * `'zero'`: Yields the Generalizes Gauss Newton matrix
        * `'abs'`: BDA-PCH approximation
        * `'clip'`: Different BDA-PCH approximation
    activation : str, 'relu' or 'sigmoid'
        Activation function
    lr : float
        Learning rate
    alpha : float, between 0 and 1
        Regularization in HVP, see Chen paper for more details
    cg_maxiter : int
        Maximum number of iterations for CG
    cg_tol : float
        Relative tolerance for convergence of CG
    cg_atol : float
        Absolute tolerance for convergence of CG
    """
    # logging directory
    # -----------------
    # directory of run
    run_name = dirname_from_params(
        act=activation,
        opt='cgn',
        batch=batch,
        lr=lr,
        alpha=alpha,
        maxiter=cg_maxiter,
        tol=cg_tol,
        atol=cg_atol,
        mod2nd=modify_2nd_order_terms)
    logdir = path.join(data_dir, run_name)

    # training procedure
    # ------------------
    def training_fn():
        """Training function setting up the train instance."""
        # set up training and run
        act = activation_dict[activation]
        model = mnist_c2d2(conv_activation=act, dense_activation=act)
        model = convert_torch_to_cvp(model)
        loss_function = convert_torch_to_cvp(CrossEntropyLoss())
        data_loader = MNISTLoader(
            train_batch_size=batch, test_batch_size=test_batch)
        optimizer = CGNewton(
            model.parameters(),
            lr=lr,
            alpha=alpha,
            cg_atol=cg_atol,
            cg_tol=cg_tol,
            cg_maxiter=cg_maxiter)
        # initialize training
        train = SecondOrderTraining(
            model,
            loss_function,
            optimizer,
            data_loader,
            logdir,
            epochs,
            modify_2nd_order_terms,
            logs_per_epoch=logs_per_epoch,
            device=device,
            input_shape=(1, 28, 28))
        return train

    return training_fn


def cgn_grid_search():
    """Define the grid search over the hyperparameters of SGD."""
    batch_sizes = [2000, 1000, 500]
    mod2nds = ['zero']
    activations = ['tanh', 'relu', 'sigmoid']
    lrs = [0.05, 0.1, 0.2]
    alphas = [0.1, 0.02, 0.01]
    cg_atol = 0.
    cg_maxiter = 100
    cg_tols = [1e-1, 1e-5]
    return [
        mnist_cgnewton_train_fn(
            batch=batch,
            modify_2nd_order_terms=mod2nd,
            activation=activation,
            lr=lr,
            alpha=alpha,
            cg_maxiter=cg_maxiter,
            cg_tol=cg_tol,
            cg_atol=cg_atol) for batch in batch_sizes for mod2nd in mod2nds
        for activation in activations for lr in lrs for alpha in alphas
        for cg_tol in cg_tols
    ]


def main(run_experiments=True):
    """Execute the experiments, return filenames of the merged runs."""
    seeds = range(1)
    labels = ['CGN-CVP']
    experiments = cgn_grid_search()

    def run():
        """Run the experiments."""
        for train_fn in experiments:
            runner = TrainingRunner(train_fn)
            runner.run(seeds)

    def result_files():
        """Merge runs and return files of the merged data."""
        filenames = OrderedDict()
        for label, train_fn in zip(labels, experiments):
            runner = TrainingRunner(train_fn)
            m_to_f = runner.merge_runs(seeds)
            filenames[label] = m_to_f
        return filenames

    if run_experiments:
        run()
    return result_files()


def filenames():
    """Return filenames of the merged data.

    Returns:
    --------
    (dict)
        A dictionary with keys given by the label of the experiments.
        The associated value itself is another dictionary with keys, values
        corresponding to the metric and filename of the metric data
        respectively.
    """
    try:
        return main(run_experiments=False)
    except Exception as e:
        print("An error occured. Maybe try to re-run the experiments.")
        raise e


if __name__ == '__main__':
    main()
