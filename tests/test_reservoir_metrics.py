"""Validate the RC metrics against known reservoir-computing theory.

Every check here uses a STANDARD tanh Echo State Network (``dynamics="esn"``) built from a
random matrix rescaled to a known spectral radius, because that is the setting in which
reservoir-computing theory makes concrete, checkable predictions (Jaeger's memory-capacity
bound, the spectral-radius-driven echo state property, the CHARC kernel/generalization-rank
trade-off). Once these hold on a textbook ESN, the same metrics can be trusted when pointed at
the physically-derived, asymmetric J from ``coilnet.physics.multibody``.
"""

import numpy as np

from coilnet.reservoir import (
    CoilReservoir,
    echo_state_property,
    generalization_rank,
    kernel_rank,
    memory_capacity,
    rescale_spectral_radius,
)

N = 40


def _build_esn(radius, gain=1.0, input_scale=1.0, noise=0.0, seed=0):
    """A random-matrix tanh ESN with a chosen spectral radius (the RC "richness" knob)."""
    rng_j = np.random.default_rng(seed)
    A = rng_j.normal(size=(N, N))
    J = rescale_spectral_radius(A, radius)
    return CoilReservoir(
        J,
        dynamics="esn",
        gain=gain,
        input_scale=input_scale,
        noise=noise,
        rng=np.random.default_rng(seed + 100),
    )


def test_memory_capacity_is_positive_and_bounded_by_N():
    """Jaeger's (2001) linear memory capacity bound: MC <= N for an N-unit reservoir."""
    res = _build_esn(radius=0.9, seed=1)
    mc = memory_capacity(res, T=1500, max_delay=30, washout=100, rng=np.random.default_rng(2))
    assert mc["MC"] > 0.0
    assert mc["MC"] <= N
    assert len(mc["per_delay"]) == 30
    assert mc["max_delay"] == 30


def test_memory_capacity_higher_near_edge_of_chaos_than_contractive():
    """A contractive reservoir (spectral radius ~0.1) forgets fast and should have markedly
    lower memory capacity than one near the edge of chaos (spectral radius ~0.9)."""
    res_rich = _build_esn(radius=0.9, seed=1)
    res_contractive = _build_esn(radius=0.1, seed=1)
    mc_rich = memory_capacity(
        res_rich, T=1500, max_delay=30, washout=100, rng=np.random.default_rng(2)
    )
    mc_contractive = memory_capacity(
        res_contractive, T=1500, max_delay=30, washout=100, rng=np.random.default_rng(2)
    )
    assert mc_rich["MC"] > mc_contractive["MC"]


def test_kernel_rank_rich_reservoir_exceeds_nearly_frozen_one():
    """A reservoir with normal gain/input scale should separate distinct inputs into
    (close to) full-rank responses; one with a tiny gain AND tiny input scale barely reacts
    to its input at all (stays in the near-linear regime around zero) and collapses most
    streams onto a low-dimensional response."""
    res_rich = _build_esn(radius=0.9, gain=1.0, input_scale=1.0, seed=3)
    res_frozen = _build_esn(radius=0.9, gain=1e-3, input_scale=1e-3, seed=3)

    kr_rich = kernel_rank(res_rich, T=200, washout=100, rng=np.random.default_rng(4))
    kr_frozen = kernel_rank(res_frozen, T=200, washout=100, rng=np.random.default_rng(4))

    assert kr_rich > kr_frozen
    assert kr_rich == N  # a well-driven 0.9-radius ESN should achieve full separation rank


def test_echo_state_property_holds_for_subunit_radius():
    res = _build_esn(radius=0.8, gain=1.0, seed=5)
    result = echo_state_property(res, T=500, washout=100, n_starts=5, rng=np.random.default_rng(6))
    assert result["has_esp"] is True
    assert result["max_distance"] < 1e-6


def test_echo_state_property_can_fail_for_large_radius_and_gain():
    res = _build_esn(radius=1.5, gain=3.0, seed=5)
    result = echo_state_property(res, T=500, washout=100, n_starts=5, rng=np.random.default_rng(6))
    assert result["has_esp"] is False
    assert result["max_distance"] > 1.0


def _build_langevin(
    radius=0.9, gain=0.5, noise=0.3, a=0.3, b=1.0, dt=0.05, input_scale=1.0, seed=11
):
    """A random-matrix BISTABLE (double-well) reservoir at the empirically-found operating
    point (see coilnet.experiments._rc_common.operating_point): enough noise to hop between
    wells, but multistable (trapped forever in whichever well it starts in) without it."""
    rng_j = np.random.default_rng(seed)
    A = rng_j.normal(size=(N, N))
    J = rescale_spectral_radius(A, radius)
    return CoilReservoir(
        J,
        dynamics="langevin",
        a=a,
        b=b,
        gain=gain,
        dt=dt,
        noise=noise,
        input_scale=input_scale,
        rng=np.random.default_rng(seed + 100),
    )


def test_echo_state_property_common_noise_needed_for_stochastic_bistable_reservoir():
    """The key empirical finding this metric fix is built around: for a bistable, noisy
    reservoir, ESP only reads correctly under "common random numbers" (common_noise=True).
    Freezing the SAME noise realization across all starts lets the shared thermal kicks erase
    the initial condition -- ESP correctly holds. With independent noise per start
    (common_noise=False), each trajectory keeps getting perturbed along its own random
    direction forever, so the final states never actually coincide -- ESP incorrectly fails,
    with a large residual tail distance -- even though the underlying substrate is the exact
    same reservoir."""
    res_common = _build_langevin()
    result_common = echo_state_property(
        res_common,
        T=1500,
        washout=500,
        n_starts=5,
        tol=0.5,
        rng=np.random.default_rng(12),
        common_noise=True,
    )
    assert result_common["has_esp"] is True
    assert result_common["max_distance"] < 1e-4

    res_indep = _build_langevin()
    result_indep = echo_state_property(
        res_indep,
        T=1500,
        washout=500,
        n_starts=5,
        tol=0.5,
        rng=np.random.default_rng(12),
        common_noise=False,
    )
    assert result_indep["has_esp"] is False
    assert result_indep["max_distance"] > 1.0
    assert result_indep["max_distance"] > result_common["max_distance"]


# ``kernel_rank``/``generalization_rank`` collapse an (n_streams, N) matrix of final states down
# to a numerical rank via ``numpy.linalg.matrix_rank``'s default (machine-epsilon-based)
# tolerance. That default tolerance is designed to catch EXACT (floating-point-roundoff-level)
# linear dependence, not "practically negligible" separation -- and a driven, continuous
# dynamical system's final states essentially never land in an EXACTLY degenerate subspace, so
# with the default tolerance the measured rank saturates near N regardless of whether the
# separation is meaningful. The tests below pass an explicit, non-default ``tol`` (chosen
# relative to the ~O(1) input scale for ``kernel_rank`` and the tiny ``perturb`` scale for
# ``generalization_rank``) so the rank actually discriminates "genuinely separated" states from
# "negligibly separated" ones -- exactly the discrimination the common-noise fix is meant to
# reveal.


def test_kernel_rank_needs_common_noise_for_noisy_bistable_reservoir():
    """The confounding bug this fix targets: without freezing the noise realization across
    streams (``common_noise=False``), every one of the N independent input streams also gets its
    own INDEPENDENT noise realization, so the final states differ from each other by noise alone
    -- the measured rank saturates near N regardless of whether the reservoir's own dynamics
    separate the inputs at all. Freezing the noise realization across streams
    (``common_noise=True``) isolates the INPUT's effect and gives a rank clearly below N for this
    bistable, noisy reservoir."""
    seed = 30
    res_indep = _build_langevin()
    kr_indep = kernel_rank(
        res_indep, T=200, washout=100, rng=np.random.default_rng(seed), common_noise=False, tol=1e-2
    )

    res_common = _build_langevin()
    kr_common = kernel_rank(
        res_common, T=200, washout=100, rng=np.random.default_rng(seed), common_noise=True, tol=1e-2
    )

    assert kr_indep >= N - 2  # saturates near full rank from noise alone
    assert kr_common <= 32  # clearly below N once the noise realization is frozen
    assert kr_common < kr_indep


def test_kernel_rank_common_noise_rich_reservoir_exceeds_nearly_frozen_one():
    """With the noise realization frozen out (``common_noise=True``), kernel rank should still
    discriminate a well-driven bistable reservoir from a nearly-frozen one (tiny gain AND tiny
    input scale barely reacts to its input at all) -- mirroring the ESN version of this test,
    but now on the physically-motivated langevin dynamics and with the noise confound removed."""
    seed = 30
    res_rich = _build_langevin()
    res_frozen = _build_langevin(gain=1e-3, input_scale=1e-3)

    kr_rich = kernel_rank(
        res_rich, T=200, washout=100, rng=np.random.default_rng(seed), common_noise=True, tol=1e-2
    )
    kr_frozen = kernel_rank(
        res_frozen, T=200, washout=100, rng=np.random.default_rng(seed), common_noise=True, tol=1e-2
    )

    assert kr_rich > kr_frozen
    assert kr_frozen <= 3  # nearly frozen: response barely separates from a single direction


def test_generalization_rank_needs_common_noise_for_noisy_bistable_reservoir():
    """Same confound, same fix, for ``generalization_rank``: without freezing the noise
    realization, the tiny (``perturb=1e-3``) input differences between streams are swamped by
    each stream's own independent noise, so the rank saturates near N (falsely reading as "poor
    generalization", i.e. amplifying noise) regardless of the reservoir's actual behaviour.
    Freezing the noise realization isolates the tiny input perturbation's effect."""
    seed = 40
    res_indep = _build_langevin()
    gr_indep = generalization_rank(
        res_indep, T=200, washout=100, rng=np.random.default_rng(seed), common_noise=False, tol=1e-4
    )

    res_common = _build_langevin()
    gr_common = generalization_rank(
        res_common, T=200, washout=100, rng=np.random.default_rng(seed), common_noise=True, tol=1e-4
    )

    assert gr_indep >= N - 2
    assert gr_common <= 20
    assert gr_common < gr_indep
