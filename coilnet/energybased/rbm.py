"""Restricted Boltzmann Machine (RBM) -- the TRAINABLE energy-based network.

``glauber.py`` asks whether a FIXED, physics-derived coupling can still do associative
recall once you let it sample instead of settle deterministically. This module asks the
complementary question: can an energy-based network on a polymer-plausible substrate
*learn* at all -- and, crucially, what happens to that learning if the weights are forced
to obey the kind of restrictions a real polymer material would impose?

An RBM is a bipartite energy-based network: visible units v (the data) and hidden units h
(latent features), with NO visible-visible or hidden-hidden connections -- only a weight
matrix W (n_visible x n_hidden) between the two layers, plus biases. Its energy is

    E(v, h) = -vᵀ W h - vbiasᵀ v - hbiasᵀ h

which makes each layer's conditional distribution factorize into independent per-unit
Bernoulli variables (the whole reason RBMs are tractable to sample and train):

    P(h_j = 1 | v) = sigmoid((v @ W)_j + hbias_j)
    P(v_i = 1 | h) = sigmoid((W @ h)_i + vbias_i)

Training uses Contrastive Divergence (CD-k, Hinton 2002): start a Gibbs chain at a real
data point, run k steps of alternating up-down sampling, and move W towards the DATA
statistics <v h>_data and away from the RECONSTRUCTION statistics <v h>_recon.

The ``weight_constraint`` hook is the point of this experiment for the materials-spec
question: it lets us re-apply a "polymer-like" restriction to W after every update, so we
can ask which restrictions a real cooperative/local/low-dimensional substrate would impose
are merely inconvenient, and which are FATAL to learning:

  * ``nonnegative`` -- all-positive weights, i.e. purely COOPERATIVE coupling (this is
    exactly what Phase 1/1b found for the PAA electrostatic coupling -- see
    ``coilnet.physics.multibody``). A Hebbian-like network can still learn with only
    excitatory connections, but an RBM's ability to represent anti-correlations (needed to
    tell APART similar-looking patterns) may be crippled.
  * ``low_rank(r)`` -- caps the number of independent coupling "modes" to r, i.e. a
    substrate with only a few independent physical degrees of freedom (few distinct
    coil-response mechanisms) rather than N^2 independently-tunable synapses.
  * ``local(bandwidth)`` -- zeros out long-range weights, i.e. a substrate where a coil can
    only influence others within some physical neighbourhood (like the short-range contact
    coupling found for the hydrophobic NVP-VPE system), not a fully-connected network.
"""

from __future__ import annotations

import numpy as np

__all__ = ["RBM", "nonnegative", "low_rank", "local"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nonnegative(W):
    """Cooperative / all-positive weight constraint (like the PAA electrostatic coupling):
    clip every weight to be >= 0."""
    return np.clip(np.asarray(W, dtype=float), 0.0, None)


def low_rank(r):
    """Factory: a constraint that projects W onto its best rank-``r`` approximation (via
    truncated SVD) after every update -- a substrate with only ``r`` independent coupling
    "modes" instead of a full, independently-tunable weight matrix."""

    def constraint(W):
        W = np.asarray(W, dtype=float)
        rank = min(r, *W.shape)
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        S_trunc = np.zeros_like(S)
        S_trunc[:rank] = S[:rank]
        return (U * S_trunc) @ Vt

    return constraint


def local(bandwidth):
    """Factory: a constraint that zeros out "long-range" weights -- visible unit i may only
    couple to hidden units within ``bandwidth`` of it, where both indices are mapped onto a
    common [0, 1] axis so the constraint is meaningful regardless of how n_visible and
    n_hidden compare. A polymer analogue of "coils can only feel neighbours nearby"."""

    def constraint(W):
        W = np.asarray(W, dtype=float)
        n_visible, n_hidden = W.shape
        vi = np.arange(n_visible)[:, None] / max(n_visible - 1, 1)
        hj = np.arange(n_hidden)[None, :] / max(n_hidden - 1, 1)
        mask = np.abs(vi - hj) <= bandwidth
        return W * mask

    return constraint


class RBM:
    """A binary (0/1 unit) Restricted Boltzmann Machine trained by contrastive divergence.

    Parameters
    ----------
    n_visible, n_hidden : int
        Layer sizes.
    rng : numpy.random.Generator, optional
        Source of randomness for initialization and sampling.
    weight_constraint : callable, optional
        If given, applied to W (returning a replacement W) right after construction and
        after every minibatch update during ``train`` -- the hook used to test
        "polymer-like" restrictions (see module docstring). Defaults to None (the
        unconstrained baseline: an ordinary, freely-trainable RBM).
    """

    def __init__(self, n_visible, n_hidden, rng=None, weight_constraint=None):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng() if rng is None else rng
        self.weight_constraint = weight_constraint
        self.W = 0.01 * self.rng.standard_normal(size=(n_visible, n_hidden))
        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)
        if self.weight_constraint is not None:
            self.W = self.weight_constraint(self.W)

    def sample_hidden(self, v):
        """P(h=1|v) and a Bernoulli sample of h. ``v`` may be a single (n_visible,) vector
        or a (batch, n_visible) matrix; the return shapes match (broadcast to 2-D)."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        prob = _sigmoid(v @ self.W + self.hbias)
        sample = (self.rng.random(prob.shape) < prob).astype(float)
        return prob, sample

    def sample_visible(self, h):
        """P(v=1|h) and a Bernoulli sample of v. Same shape convention as sample_hidden."""
        h = np.atleast_2d(np.asarray(h, dtype=float))
        prob = _sigmoid(h @ self.W.T + self.vbias)
        sample = (self.rng.random(prob.shape) < prob).astype(float)
        return prob, sample

    def free_energy(self, v):
        """F(v) = -vbiasᵀv - sum_j softplus((v@W)_j + hbias_j) -- the hidden units marginalized
        out analytically. Lower free energy = the model considers v more likely; used mainly
        as a training-progress diagnostic (it should trend down for the training data)."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        wx_b = v @ self.W + self.hbias
        hidden_term = np.sum(np.logaddexp(0.0, wx_b), axis=1)
        visible_term = v @ self.vbias
        result = -visible_term - hidden_term
        return result

    def reconstruct(self, v):
        """One up-down pass v -> P(h|v) -> P(v|h). Returns the reconstruction PROBABILITY
        (not a resampled binary vector) -- the network's best continuous guess at v."""
        v = np.atleast_2d(np.asarray(v, dtype=float))
        _, h_sample = self.sample_hidden(v)
        v_prob, _ = self.sample_visible(h_sample)
        return v_prob

    def train(self, data, epochs, lr=0.1, k=1, batch_size=10):
        """Contrastive Divergence (CD-k) training (Hinton, 2002).

        For each minibatch: sample h0 from the real data v0, run k steps of alternating
        Gibbs sampling (v -> h -> v -> ...) to get a "fantasy" (vk, hk), then update

            W      += lr * (v0ᵀ h0_prob - vkᵀ hk_prob) / batch_size
            vbias  += lr * mean(v0 - vk, axis=0)
            hbias  += lr * mean(h0_prob - hk_prob, axis=0)

        If ``self.weight_constraint`` is set, it is RE-APPLIED to W after every minibatch --
        this is how we test whether a "polymer-like" restriction on the weights is even
        compatible with learning, not just with the initial random W.

        Returns a list of per-epoch mean-squared reconstruction errors (one-step v ->
        reconstruct(v) error on that epoch's data) -- should fall clearly for an
        unconstrained network if CD-k learning is working at all.
        """
        data = np.atleast_2d(np.asarray(data, dtype=float))
        n = len(data)
        history = []
        for _epoch in range(epochs):
            order = self.rng.permutation(n)
            errors = []
            for start in range(0, n, batch_size):
                batch = data[order[start : start + batch_size]]
                b = len(batch)

                v0 = batch
                h0_prob, h0_sample = self.sample_hidden(v0)

                vk_sample, hk_sample = v0, h0_sample
                hk_prob = h0_prob
                for _step in range(k):
                    _, vk_sample = self.sample_visible(hk_sample)
                    hk_prob, hk_sample = self.sample_hidden(vk_sample)

                self.W += lr * (v0.T @ h0_prob - vk_sample.T @ hk_prob) / b
                self.vbias += lr * np.mean(v0 - vk_sample, axis=0)
                self.hbias += lr * np.mean(h0_prob - hk_prob, axis=0)

                if self.weight_constraint is not None:
                    self.W = self.weight_constraint(self.W)

                recon = self.reconstruct(v0)
                errors.append(float(np.mean((v0 - recon) ** 2)))
            history.append(float(np.mean(errors)))
        return history

    def gibbs_sample(self, n_samples, n_steps):
        """Generate ``n_samples`` binary visible vectors by running ``n_steps`` of block
        Gibbs sampling (v -> h -> v) from random initial states, one independent chain per
        sample -- what the trained network has "learned to generate", the RBM analogue of
        Hopfield recall from a blank slate rather than from a cue."""
        v = (self.rng.random((n_samples, self.n_visible)) < 0.5).astype(float)
        for _ in range(n_steps):
            _, h = self.sample_hidden(v)
            _, v = self.sample_visible(h)
        return v
