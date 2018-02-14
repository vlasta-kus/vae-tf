from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf

from layers import Dense
import plot
from utils import composeAll, print_


class VAE():
    """Variational Autoencoder

    see: Kingma & Welling - Auto-Encoding Variational Bayes
    (http://arxiv.org/abs/1312.6114)
    """
    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.,
        "lambda_l2_reg": 0.,
        "nonlinearity": tf.nn.elu,
        "output_activation": tf.nn.sigmoid
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 save_graph_def=True, log_dir="./log"):
        """(Re)build a symmetric VAE model with given:

            * architecture (list of nodes per encoder layer); e.g.
               [1000, 500, 250, 10] specifies a VAE with 1000-D inputs, 10-D latents,
               & end-to-end architecture [1000, 500, 250, 10, 250, 500, 1000]

            * hyperparameters (optional dictionary of updates to `DEFAULTS`)
        """
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if not meta_graph: # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers! (input, 1+ hidden, latent)"

            # build graph
            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.merged_summaries = tf.summary.merge_all() # merge all the summaries and write them out
            self.sesh.run(tf.global_variables_initializer())

        else: # restore saved model
            print("ERROR: not supported right now -> Vlasta, upgrade vae.py!")
            #model_datetime, model_name = os.path.basename(meta_graph).split("_vae_")
            #self.datetime = "{}_reloaded".format(model_datetime)
            #*model_architecture, _ = re.split("_|-", model_name)
            #self.architecture = [int(n) for n in model_architecture]

            # rebuild graph
            #meta_graph = os.path.abspath(meta_graph)
            #tf.train.import_meta_graph(meta_graph + ".meta").restore(
            #    self.sesh, meta_graph)
            #handles = self.sesh.graph.get_collection(VAE.RESTORE_KEY)

        # unpack handles for tensor ops to feed or fetch
        (self.x_in, self.dropout_prob, self.z_mean, self.z_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.total_updates, self.train_op) = handles

        if save_graph_def: # tensorboard
            self.logger = tf.summary.FileWriter(log_dir, self.sesh.graph)

    @property
    def step(self):
        """Train step"""
        return self.total_updates.eval(session=self.sesh)

    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, # enables variable batch size
                                                 self.architecture[0]], name="x")
        dropout_prob = tf.placeholder_with_default(1., shape=[], name="dropout_prob")

        # encoding / "recognition": q(z|x)
        encoding = [Dense("encoding", idx, hidden_size, dropout_prob, self.nonlinearity) for hidden_size, idx in zip(self.architecture[1:-1], range(len(self.architecture) - 2))]
        # hidden layers reversed for function composition: outer -> inner
        h_encoded = composeAll(reversed(encoding))(x_in)
        self.layerSummaries(encoding, x_in)

        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, z_sigma**2)
        z_mean = Dense("z_mean", len(self.architecture) - 2, self.architecture[-1], 1.)(h_encoded)
        z_sigma = Dense("z_sigma", len(self.architecture) - 2, self.architecture[-1], 1.)(h_encoded)
        tf.summary.histogram('activations_z_mean', z_mean)
        tf.summary.histogram('activations_z_sigma', z_sigma)
        tf.summary.scalar('z_mean_mean-activations', tf.reduce_mean(z_mean))
        tf.summary.scalar('z_sigma_mean-activations', tf.reduce_mean(z_sigma))

        # kingma & welling: only 1 draw necessary as long as minibatch large enough (>100)
        z = self.sampleGaussian(z_mean, z_sigma)
        tf.summary.histogram('z_sampled', z)

        # decoding (p(x|z)): assuming architecture symmetry
        decoding = [Dense("decoding", idx, hidden_size, dropout_prob, self.nonlinearity)
                        for hidden_size, idx in zip(self.architecture[1:-1], reversed(range(len(self.architecture)-1, 2*len(self.architecture) - 3, 1)))]
        # final reconstruction: prepend as outermost function
        decoding.insert(0, Dense("x_decoding", 2*len(self.architecture) - 3, self.architecture[0], 1.0, self.output_activation))
        x_reconstructed = tf.identity(composeAll(decoding)(z), name="x_reconstructed")
        self.layerSummaries(reversed(decoding), z)

        # reconstruction loss: mismatch b/w x & x_reconstructed
        # binary cross-entropy -- assumes x & p(x|z) are iid Bernoullis
        rec_loss = VAE.crossEntropy(x_reconstructed, x_in)
        #rec_loss = VAE.l2_loss(x_reconstructed, x_in);
        tf.summary.scalar('cross_entropy', tf.reduce_mean(rec_loss))

        # Kullback-Leibler divergence: mismatch b/w approximate vs. imposed/true posterior
        kl_loss = VAE.kullbackLeibler(z_mean, z_sigma)
        tf.summary.scalar('KL_divergence', tf.reduce_mean(kl_loss))

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection("trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)
        tf.summary.scalar('l2_reg', l2_reg)

        with tf.name_scope("cost"):
            # average over minibatch
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            tf.summary.scalar('cost_without_regularization', cost)
            cost += l2_reg
        tf.summary.scalar('cost', cost)

        # optimization
        total_updates = tf.Variable(0, trainable=False)
        #with tf.name_scope("Adam_optimizer"):
        with tf.name_scope("Adagrad_optimizer"):
            #optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar) # gradient clipping
                    for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=total_updates, name="minimize_cost")

        # ops to directly explore latent space
        # defaults to prior z ~ N(0, I)
        with tf.name_scope("latent_in"):
            z_ = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]),
                                            shape=[None, self.architecture[-1]],
                                            name="latent_in")
        x_reconstructed_ = composeAll(decoding)(z_)

        return (x_in, dropout_prob, z_mean, z_sigma, x_reconstructed, z_, x_reconstructed_, cost, total_updates, train_op)

    def sampleGaussian(self, mu, sigma):
        """(Differentiably!) draw sample from Gaussian with given shape, subject to random noise epsilon"""
        with tf.name_scope("sample_gaussian"):
            # reparameterization trick
            epsilon = tf.random_normal(tf.shape(sigma), name="epsilon")
            return mu + epsilon * sigma # N(mu, I * sigma**2)

    @staticmethod
    def crossEntropy(obs, actual, offset=1e-7):
        """Binary cross-entropy, per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return -tf.reduce_sum(actual * tf.log(obs_) + (1 - actual) * tf.log(1 - obs_), 1)

    @staticmethod
    def l1_loss(obs, actual):
        """L1 loss (a.k.a. LAD), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l1_loss"):
            return tf.reduce_sum(tf.abs(obs - actual) , 1)

    @staticmethod
    def l2_loss(obs, actual):
        """L2 loss (a.k.a. Euclidean / LSE), per training example"""
        # (tf.Tensor, tf.Tensor, float) -> tf.Tensor
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullbackLeibler(mu, sigma):
        """(Gaussian) Kullback-Leibler divergence KL(q||p), per training example"""
        with tf.name_scope("KL_divergence"):
            return 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(tf.square(sigma)) - 1, 1)

    def encode(self, x):
        """Probabilistic encoder from inputs to latent distribution parameters;
        a.k.a. inference network q(z|x)
        """
        # np.array -> [float, float]
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        """Generative decoder from latent space to reconstructions of input space;
        a.k.a. generative network p(x|z)
        """
        # (np.array | tf.Variable) -> np.array
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sesh.run(zs) if is_tensor(zs) else zs) # coerce to np.array
            feed_dict.update({self.z_: zs})
        # else, zs defaults to draw from conjugate prior z ~ N(0, I)
        return self.sesh.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        """End-to-end autoencoder"""
        # np.array -> np.array
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def layerSummaries(self, layers, input):
        previous_out = input
        for layer in layers:
            curr_out = layer(previous_out)
            mean = tf.reduce_mean(curr_out)
            stddev = tf.sqrt(tf.reduce_mean( tf.square(curr_out - mean) ))
            tf.summary.histogram(layer.name + '_activations', curr_out)
            tf.summary.scalar(layer.name + '_mean-activations', mean)
            tf.summary.scalar(layer.name + '_stddev-activations', stddev)
            previous_out = curr_out

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True, verbose=True, save=True, outdir="./out", plots_outdir="./png", plot_latent_over_time=False):
        if save:
            saver = tf.train.Saver(tf.global_variables())

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            if plot_latent_over_time: # plot latent space over log_BASE time
                BASE = 2
                INCREMENT = 0.5
                pow_ = 0

            nBatches = 0
            while True:
                nBatches += 1
                x, _ = X.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.dropout_prob: self.dropout}
                fetches = [self.x_reconstructed, self.cost, self.total_updates, self.train_op, self.merged_summaries]
                x_reconstructed, cost, i, _, summary = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if plot_latent_over_time:
                    while int(round(BASE**pow_)) == i:
                        plot.exploreLatent(self, nx=30, ny=30, ppf=True, outdir=plots_outdir,
                                           name="explore_ppf30_{}".format(pow_))

                        names = ("train", "validation", "test")
                        datasets = (X.train, X.validation, X.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self, dataset.images, dataset.labels, range_=
                                              (-6, 6), title=name, outdir=plots_outdir,
                                              name="{}_{}".format(name, pow_))

                        print("{}^{} = {}".format(BASE, pow_, i))
                        pow_ += INCREMENT

                if i%10 == 0:
                    #run_metadata = tf.RunMetadata()
                    #self.logger.add_run_metadata(run_metadata, 'step%03d' % i)
                    self.logger.add_summary(summary, i)

                if i%50 == 0 and verbose:
                    print("  iteration {} --> current cost: {}".format(i, cost))

                if i%1000 == 0 and verbose:
                    print("  \titeration {} --> total avg cost: {}".format(i, err_train / i))

                if i%1000 == 0 and verbose: # and i >= 10000:
                    # visualize `n` examples of current minibatch inputs + reconstructions
                    plot.plotSubset(self, x, x_reconstructed, n=10, name="train",
                                    outdir=plots_outdir)

                    if cross_validate:
                        x, _ = X.validation.next_batch(self.batch_size)
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed, self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches, feed_dict)

                        print("  iteration {} --> CV cost: ".format(i), cost)
                        plot.plotSubset(self, x, x_reconstructed, n=10, name="cv",
                                        outdir=plots_outdir)

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
                            self.datetime, "_".join(map(str, self.architecture))))
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        continue
                    break
            print(" >>> Processed %d batches of size %d, i.e. %d data samples." % (nBatches, self.batch_size, nBatches * self.batch_size))

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)
