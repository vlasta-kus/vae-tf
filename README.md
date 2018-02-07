# vae-ry exciting vae code
For all your TensorFlow Variational Autoencoder needs.

## Description

This is the repository for the Variational Autoencoder (VAE) blogpost series
from [Fast Forward Labs](http://www.fastforwardlabs.com).
Start there, then check out the repo!

* [Part I: Introducing Variational Autoencoders (in Prose and Code)](http://blog.fastforwardlabs.com/2016/08/12/introducing-variational-autoencoders-in-prose-and.html)
* [Part II: Under the Hood of the Variational Autoencoder (in Prose and Code)](http://blog.fastforwardlabs.com/2016/08/22/under-the-hood-of-the-variational-autoencoder-in.html)

## Usage:

To train a new model, edit `main.py` with your desired VAE `ARCHITECTURE`,
`HYPERPARAMETERS`, and `paths/to/outdirs`.

Then, simply:

```
$ python main.py
```

OR, restore a trained model from its saved `meta_graph` via:

```
$ python main.py <path/to/meta_graph_name>
```
(without the `.meta` suffix)
