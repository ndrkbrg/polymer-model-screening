"""Phase 1.8 -- two routes from a WORKING energy-based (Boltzmann) network to a PHYSICAL one.

Phase 1.7 found that an energy-based network (``coilnet.energybased.rbm.RBM``) learns well and
survives most polymer-like *structural* restrictions on W (cooperative-only, short-range,
modest low-rank) -- but flagged **trainability** as the single hardest, still-open requirement:
a real polymer's coupling is set by chemistry and geometry, not obviously adjustable at will.
This module builds the two concrete ways a physical substrate could still end up with a
*working* coupling despite that, so they can be compared head to head:

  * **Route B -- offline train + fabricate.** Train the network ideally, in software, where
    weight updates are exact and unconstrained; then *build* the resulting W once, in a real
    material. The physical build step cannot be perfect: every coupling comes out multiplied by
    some random relative error, and/or snapped onto a coarse grid of physically realizable
    values (a polymer only has so many distinguishable crosslink densities / dopant levels /
    whatever "sets" a given coupling strength). ``fabricate`` models exactly this: one-shot
    noise + quantization applied to an already-trained W. It never touches the learning
    dynamics -- by the time it runs, training is over.

  * **Route A -- local-rule in-place learning.** Skip fabrication entirely and let the material
    learn where it sits, using a LOCAL rule (contrastive divergence is already local: each
    weight update dW_ij depends only on units i and j, never on remote weights) -- but a real
    local plasticity mechanism is itself degraded relative to the software ideal. Two very
    different kinds of degradation are modeled here, and they give OPPOSITE verdicts:

      - BENIGN, unbiased imperfections: noisy (thermal/compositional variability in how much a
        given "nudge" actually changes the coupling), quantized (physical plasticity happens in
        discrete steps -- a fixed crosslinking/doping increment, not an arbitrary real number),
        and a constant sign-asymmetry (a fixed ``potentiation_ratio`` scaling every depressing
        update by the same factor). These self-average over thousands of updates, so Route A
        looks deceptively robust to them.
      - REALISTIC memristive plasticity: every real memristive/adaptive device's update is
        NONLINEAR and STATE-DEPENDENT (how much a nudge changes the coupling depends on where
        the coupling currently sits in its range), SATURATING (potentiation shrinks to zero as
        the weight approaches its positive limit, depression shrinks to zero as it approaches
        its negative limit), and ASYMMETRIC (depression is intrinsically weaker than
        potentiation, not just scaled by a constant). Unlike the benign case, this does NOT
        self-average -- it systematically biases the whole learning trajectory towards whatever
        the device's fixed points are, and collapses learning (verified: BAS validity drops from
        ~0.96 ideal to ~0.11 at ``mem_gamma=2.0, mem_dep_ratio=0.5`` on Bars-and-Stripes size 3).

    ``train_degraded_local`` reimplements the RBM's CD-k update loop with all of the above
    available as opt-in degradations applied to dW *during* training, so imperfection
    accumulates over the whole learning trajectory rather than being applied once at the end.

These are NOT apples-to-apples in one respect, and the run script says so explicitly: Route A
perturbs the *learning path* (many small degraded steps, whose errors can compound or partially
cancel over thousands of updates) while Route B perturbs the *final weights* (one degradation,
applied once). Comparing them still answers the practically important question -- "given some
budget of physical imprecision, which route gets you a better and more reproducible network" --
it just is not a controlled comparison of a single shared noise model. The memristive-plasticity
result sharpens this further: it shows the "which route is easier" answer depends critically on
whether Route A's local plasticity is modeled as benign noise (looks fine) or as the nonlinear,
state-dependent, saturating, asymmetric behavior real memristive devices actually exhibit
(collapses) -- see ``coilnet.experiments.run_phase18_routes`` for the corrected verdict.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fabricate", "train_degraded_local"]


def _quantize(X, n_levels):
    """Snap every entry of ``X`` onto one of ``n_levels`` uniformly-spaced values spanning
    ``X``'s own [min, max] range. Shared by ``fabricate`` (quantizing a trained W) and
    ``train_degraded_local`` (quantizing each minibatch's dW) -- both model a physical
    substrate that can only realize a coarse grid of values, sized to whatever range of values
    is actually in play (a trained W's overall scale, or one update's own dynamic range) rather
    than some externally-fixed absolute unit.

    Returns at most ``n_levels`` distinct values. If ``X`` is (numerically) constant, every
    entry maps to that constant -- there is nothing to quantize.
    """
    X = np.asarray(X, dtype=float)
    if n_levels is None or n_levels < 1:
        return X.copy()
    lo, hi = float(X.min()), float(X.max())
    if hi <= lo or n_levels == 1:
        return np.full_like(X, lo)
    idx = np.round((X - lo) / (hi - lo) * (n_levels - 1))
    idx = np.clip(idx, 0, n_levels - 1)
    return lo + idx * (hi - lo) / (n_levels - 1)


def fabricate(W, rel_noise=0.0, n_levels=None, rng=None):
    """Route B: apply ONE-SHOT physical fabrication error to an already-trained coupling ``W``.

    Models "you can't build the exact target coupling you computed in software -- only a
    noisy, coarsely-quantized version of it":

      1. Multiplicative noise: each weight is scaled by ``(1 + rel_noise * N(0, 1))`` --
         relative, not absolute, error (a bigger weight suffers a bigger absolute error), the
         natural way to model imprecise fabrication of a continuous physical quantity.
      2. Quantization (optional): if ``n_levels`` is given, the noised weights are then snapped
         onto ``n_levels`` uniformly-spaced values spanning their own min/max (see
         ``_quantize``) -- a substrate that can only realize a finite palette of coupling
         strengths.

    With ``rel_noise=0`` and ``n_levels=None`` this is the identity (returns ``W`` unchanged) --
    the "perfect fabrication" limit used as a sanity check.

    Parameters
    ----------
    W : array_like
        The trained weight matrix (or any array) to fabricate.
    rel_noise : float
        Standard deviation of the *relative* per-weight fabrication error. 0 = perfect.
    n_levels : int, optional
        Number of physically distinguishable coupling levels. None = continuous (no
        quantization).
    rng : numpy.random.Generator, optional
        Source of the fabrication noise.

    Returns
    -------
    ndarray
        The fabricated (perturbed) weight matrix. ``W`` itself is not modified.
    """
    rng = np.random.default_rng() if rng is None else rng
    W = np.asarray(W, dtype=float)
    fabricated = W * (1.0 + rel_noise * rng.standard_normal(size=W.shape))
    if n_levels is not None:
        fabricated = _quantize(fabricated, n_levels)
    return fabricated


def _degrade_dW(dW, update_noise, update_levels, potentiation_ratio, rng):
    """Apply the three DURING-LEARNING degradations to one minibatch's weight update ``dW``
    (physical saturation/clipping is applied to W itself, after the update -- see
    ``train_degraded_local``):

      (a) relative Gaussian noise, scaled to each entry's own |dW| (a bigger intended nudge
          suffers a bigger absolute error -- same relative-error logic as ``fabricate``);
      (b) quantization onto ``update_levels`` discrete step sizes spanning this batch's own dW
          range (a local plasticity mechanism that can only change a synapse by one of a
          handful of fixed physical increments, not an arbitrary real-valued nudge -- note this
          can occasionally bias the sign of small updates within a batch, which is part of the
          degradation being modeled, not a bug);
      (c) sign-asymmetric plasticity: the POTENTIATING (positive) part of dW is left alone,
          the DEPRESSING (negative) part is scaled by ``potentiation_ratio`` -- a
          ``potentiation_ratio < 1`` means depression is physically weaker than potentiation
          (the common real-memristor asymmetry).
    """
    dW = np.asarray(dW, dtype=float).copy()
    if update_noise > 0:
        dW = dW + update_noise * np.abs(dW) * rng.standard_normal(size=dW.shape)
    if update_levels is not None:
        dW = _quantize(dW, update_levels)
    if potentiation_ratio != 1.0:
        dW = np.clip(dW, 0.0, None) + potentiation_ratio * np.clip(dW, None, 0.0)
    return dW


def _memristive_dW(dW, W, wmax, gamma, dep_ratio):
    """Apply the VERIFIED realistic-memristive-plasticity transform to one minibatch's weight
    update ``dW``, given the CURRENT weights ``W`` (i.e. before this update is applied).

    Unlike ``_degrade_dW``'s benign, unbiased degradations (which self-average over many
    updates), this models how every real memristive/adaptive device actually updates: the
    EFFECTIVE step size depends on where each weight currently sits in its own [-wmax, +wmax]
    range, not just on the intended dW --

      - potentiation (dW > 0, strengthening a coupling) shrinks towards zero as the weight
        approaches +wmax -- a device running out of room to potentiate further;
      - depression (dW < 0, weakening a coupling) is scaled by ``dep_ratio`` (< 1 means
        depression is intrinsically WEAKER than potentiation, the common real-memristor
        asymmetry) and shrinks towards zero as the weight approaches -wmax;
      - ``gamma`` controls how sharply each effect turns on: gamma=1 is a mild linear-saturating
        rolloff, larger gamma concentrates the loss of plasticity closer to the rails.

    This is a STATE-DEPENDENT, NONLINEAR bias, not noise -- it does not average out over epochs,
    it systematically drags training towards whatever fixed point the device's asymmetric
    saturation implies, which is why it collapses learning where the benign degradations above do
    not (verified: BAS validity ~0.96 -> ~0.11 at gamma=2.0, dep_ratio=0.5 on Bars-and-Stripes
    size 3 -- see the module docstring and ``run_phase18_routes``).
    """
    dW = np.asarray(dW, dtype=float)
    W = np.asarray(W, dtype=float)
    frac = np.clip((W + wmax) / (2.0 * wmax), 0.0, 1.0)
    pot = np.maximum(0.0, 1.0 - frac) ** gamma
    dep = dep_ratio * np.maximum(0.0, frac) ** gamma
    return np.where(dW > 0, dW * pot, dW * dep)


def train_degraded_local(
    rbm,
    data,
    epochs,
    lr=0.5,
    k=1,
    batch_size=14,
    update_noise=0.0,
    update_levels=None,
    potentiation_ratio=1.0,
    w_max=None,
    memristive=False,
    mem_gamma=2.0,
    mem_dep_ratio=0.5,
    mem_wmax=2.5,
    rng=None,
):
    """Route A: train ``rbm`` IN PLACE with a physically-DEGRADED LOCAL learning rule.

    This reimplements ``RBM.train``'s contrastive-divergence (CD-k) loop exactly (same
    Gibbs-sampling update, same per-minibatch structure) -- it is already local, since each
    dW_ij depends only on units i and j -- but degrades the weight update dW each minibatch
    before applying it, then optionally clips |W| (physical saturation of the coupling). Two
    independent, separately-toggled degradation stories are available (see the module
    docstring for why they give OPPOSITE verdicts):

      1. BENIGN degradations, via ``_degrade_dW`` (noise, quantization, a constant
         sign-asymmetric ``potentiation_ratio``), then a flat ``w_max`` clip. These are
         unbiased and self-average over many updates.
      2. REALISTIC memristive plasticity, via ``_memristive_dW`` (``memristive=True``): a
         nonlinear, STATE-DEPENDENT, saturating, asymmetric transform of dW based on where each
         weight currently sits in its own [-mem_wmax, +mem_wmax] range, followed by a
         ``mem_wmax`` clip. This does NOT self-average and collapses learning at realistic
         settings.

    Both can be combined (the benign degradations applied first, then the memristive
    transform), but they are normally swept independently. With every degradation left at its
    ideal/off default (``update_noise=0``, ``update_levels=None``, ``potentiation_ratio=1.0``,
    ``w_max=None``, ``memristive=False``) this reduces to ordinary CD-k -- the "perfect local
    learning" limit used as a sanity check.

    The BIASES (vbias, hbias) are updated with the ordinary, un-degraded rule. This is a
    deliberate simplification, not an oversight: the physically interesting, "is this coupling
    even buildable" question is about the coupling W, and giving the biases a free pass avoids
    conflating two independent degradation stories in one set of sweeps (a bias is also a much
    lower-dimensional, easier-to-get-right quantity than an N_visible x N_hidden coupling
    matrix).

    Parameters
    ----------
    rbm : RBM
        Trained in place; also returned for convenience.
    data, epochs, lr, k, batch_size : as in ``RBM.train``.
    update_noise : float
        Relative noise (see ``_degrade_dW``) added to every minibatch's dW. 0 = no noise.
    update_levels : int, optional
        Number of discrete physical update-step sizes. None = continuous dW.
    potentiation_ratio : float
        Multiplier on the DEPRESSING (negative) part of dW; 1.0 = symmetric/ideal.
    w_max : float, optional
        If given, |W| is clipped to this value after every minibatch update (physical
        saturation). None = unbounded.
    memristive : bool
        If True, apply the REALISTIC memristive-plasticity transform (``_memristive_dW``) to
        dW every minibatch, using the weights' CURRENT position in [-mem_wmax, mem_wmax], and
        clip |W| to ``mem_wmax`` after every update. False (default) = the benign-degradation
        path above only, unchanged.
    mem_gamma : float
        Sharpness of the memristive saturation rolloff (see ``_memristive_dW``). Only used
        when ``memristive=True``.
    mem_dep_ratio : float
        Memristive depression-vs-potentiation asymmetry ratio (see ``_memristive_dW``). Only
        used when ``memristive=True``.
    mem_wmax : float
        Memristive device's coupling range, [-mem_wmax, mem_wmax]. Only used when
        ``memristive=True``.
    rng : numpy.random.Generator, optional
        Source of the degradation noise. Defaults to ``rbm.rng`` (so a single seeded RBM fully
        determines the run) if not given separately.

    Returns
    -------
    RBM
        The same ``rbm`` object, trained in place.
    """
    rng = rbm.rng if rng is None else rng
    data = np.atleast_2d(np.asarray(data, dtype=float))
    n = len(data)
    for _epoch in range(epochs):
        order = rbm.rng.permutation(n)
        for start in range(0, n, batch_size):
            batch = data[order[start : start + batch_size]]
            b = len(batch)

            v0 = batch
            h0_prob, h0_sample = rbm.sample_hidden(v0)

            vk_sample, hk_sample = v0, h0_sample
            hk_prob = h0_prob
            for _step in range(k):
                _, vk_sample = rbm.sample_visible(hk_sample)
                hk_prob, hk_sample = rbm.sample_hidden(vk_sample)

            dW = (v0.T @ h0_prob - vk_sample.T @ hk_prob) / b
            dW = _degrade_dW(dW, update_noise, update_levels, potentiation_ratio, rng)
            if memristive:
                dW = _memristive_dW(dW, rbm.W, mem_wmax, mem_gamma, mem_dep_ratio)

            rbm.W += lr * dW
            rbm.vbias += lr * np.mean(v0 - vk_sample, axis=0)
            rbm.hbias += lr * np.mean(h0_prob - hk_prob, axis=0)

            if w_max is not None:
                np.clip(rbm.W, -w_max, w_max, out=rbm.W)
            if memristive:
                np.clip(rbm.W, -mem_wmax, mem_wmax, out=rbm.W)

    return rbm
