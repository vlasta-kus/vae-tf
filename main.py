import os
import sys
import collections

import numpy as np
import tensorflow as tf

import plot
from utils import get_mnist

import vae
from DataFromTxt import DataFromTxt


IMG_DIM = 28

# Encoder architecture (decoder is symmetric)
#ARCHITECTURE = [IMG_DIM**2, # 784 pixels
#                #500, 500, # intermediate encoding
#                #2] # latent space dims
#                400, 200,
#                20]

ARCHITECTURE = [60, 40, 20, 10]

HYPERPARAMS = {
    #"batch_size": 128,
    "batch_size": 10,
    "learning_rate": 0.01,
    "dropout": 0.9,
    "lambda_l2_reg": 1E-5,
    #"nonlinearity": tf.nn.elu,
    "nonlinearity": tf.nn.relu,
    "output_activation": tf.nn.sigmoid
    #"output_activation": tf.identity
}

MAX_ITER = 3000 #2**16
MAX_EPOCHS = 20 #np.inf

LOG_DIR = "./log"
METAGRAPH_DIR = "./out"
PLOTS_DIR = "./png"


def load_mnist():
    from tensorflow.examples.tutorials.mnist import input_data
    return input_data.read_data_sets("./mnist_data")

def load_textual_data(path):
    Datasets = collections.namedtuple('Datasets', ['train', 'validation', 'test'])
    data = DataFromTxt(path)
    return Datasets(train=data, validation=None, test=None)

def all_plots(model, mnist):
    if model.architecture[-1] == 2: # only works for 2-D latent
        print("Plotting in latent space...")
        plot_all_in_latent(model, mnist)

        print("Exploring latent...")
        plot.exploreLatent(model, nx=20, ny=20, range_=(-4, 4), outdir=PLOTS_DIR)
        for n in (24, 30): #, 60, 100):
            plot.exploreLatent(model, nx=n, ny=n, ppf=True, outdir=PLOTS_DIR, name="explore_ppf{}".format(n))

    #print("Interpolating...")
    #interpolate_digits(model, mnist)

    print("Plotting end-to-end reconstructions...")
    plot_all_end_to_end(model, mnist)

    print("Morphing...")
    morph_numbers(model, mnist, ns=[9,8,7,6,5,4,3,2,1,0], n_per_morph=5)

    #print("Plotting 10 MNIST digits...")
    #for i in range(10):
    #    plot.justMNIST(get_mnist(i, mnist), name=str(i), outdir=PLOTS_DIR)

def plot_all_in_latent(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        plot.plotInLatent(model, dataset.images, dataset.labels, name=name, outdir=PLOTS_DIR)

def interpolate_digits(model, mnist):
    imgs, labels = mnist.train.next_batch(100)
    idxs = np.random.randint(0, imgs.shape[0] - 1, 2)
    mus, _ = model.encode(np.vstack(imgs[i] for i in idxs))
    plot.interpolate(model, *mus, name="interpolate_{}->{}".format(
        *(labels[i] for i in idxs)), outdir=PLOTS_DIR)

def plot_all_end_to_end(model, mnist):
    names = ("train", "validation", "test")
    datasets = (mnist.train, mnist.validation, mnist.test)
    for name, dataset in zip(names, datasets):
        x, _ = dataset.next_batch(10)
        x_reconstructed = model.vae(x)
        plot.plotSubset(model, x, x_reconstructed, n=10, name=name, outdir=PLOTS_DIR)

def morph_numbers(model, mnist, ns=None, n_per_morph=10):
    if not ns:
        import random
        ns = random.sample(range(10), 10) # non-in-place shuffle

    xs = np.squeeze([get_mnist(n, mnist) for n in ns])
    mus, _ = model.encode(xs)
    plot.morph(model, mus, n_per_morph=n_per_morph, outdir=PLOTS_DIR, name="morph_{}".format("".join(str(n) for n in ns)))

def main(data="mnist", to_reload=None):
    if data == "mnist":
        input_data = load_mnist()
        control_plots = True
    elif data == "sentences":
        input_data = load_textual_data("data/sentenceVectors-Emails-January.out")
        control_plots = False;

    if to_reload: # restore
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, meta_graph=to_reload)
        print("Loaded!")

    else: # train
        v = vae.VAE(ARCHITECTURE, HYPERPARAMS, log_dir=LOG_DIR)
        v.train(input_data, max_iter=MAX_ITER, max_epochs=MAX_EPOCHS, cross_validate=False,
                verbose=True, save=True, outdir=METAGRAPH_DIR, plots_outdir=PLOTS_DIR,
                plot_latent_over_time=False, control_plots=control_plots)
        print("Trained!\n")

    #all_plots(v, input_data)


if __name__ == "__main__":
    alg = "mnist"
    if len(sys.argv) > 1:
        alg = sys.argv[1]

    tf.reset_default_graph()

    for DIR in (LOG_DIR, METAGRAPH_DIR, PLOTS_DIR):
        try:
            os.mkdir(DIR)
        except:
            pass

    main(alg, None);

    #try:
    #    to_reload = sys.argv[1]
    #    main(to_reload=to_reload, alg)
    #except(IndexError):
    #    main()

