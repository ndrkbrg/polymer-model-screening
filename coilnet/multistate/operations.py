"""Phase 1.6c -- register vs. PROCESSOR: can control inputs drive reliable, STATE-DEPENDENT
transitions between the alphabet states, not just load/reset them to a fixed target?

``addressability.py`` showed a chosen state can be WRITTEN and HELD -- the substrate is a
controllable multi-state REGISTER. That is necessary but not sufficient for computation. The
crucial distinction this module tests:

  * A control input that drives EVERY alphabet state to the SAME target is a "load/reset"
    (``constant_load``): the output does not depend on the operand at all. That is storage, not
    computation.
  * A control input whose output DEPENDS on which state it started from is a genuine GATE. The
    gold standard is a PERMUTATION of the alphabet (a reversible, state-dependent bijection,
    ``permutation``) -- and, best of all, a clean k-CYCLE (a single cycle through all k states),
    which is exactly an INCREMENT: the generator of k-valued cyclic arithmetic that Papers 4-5's
    multi-valued logic proposal ultimately needs.

The protocol, built directly on ``coilnet.hopfield.HopfieldNetwork`` and the alphabet of
addressable states from ``addressability.py`` / ``landscape.py``:

  1. Relax EVERY alphabet state under a bias field u (``HopfieldNetwork(J, bias=u)``) -- this is
     "applying a control input" (``induced_map``).
  2. Snap each relaxed result to the NEAREST alphabet state by Hamming distance. If it lands
     cleanly (small "leak"), the field induces a well-defined map on the alphabet; classify that
     map as identity / constant_load / permutation / function / leaky (``classify_map``).
  3. Search many random control fields, at several field strengths (as fractions of the
     coupling's own natural scale, ``field_scale`` -- the pilot showed field strength is
     decisive: too weak gives identity, too strong gives constant_load, gates if any live in
     between) for state-dependent gates (``gate_search``), for a clean k-cycle / increment
     (``try_increment``), and for how many times a discovered gate can be CHAINED before its
     output derails from the ideal iterated map (``composability``).
"""

from __future__ import annotations

import numpy as np

from ..hopfield import HopfieldNetwork

__all__ = [
    "field_scale",
    "induced_map",
    "classify_map",
    "largest_cycle_length",
    "gate_search",
    "try_increment",
    "composability",
]

MAP_CLASSES = ("identity", "constant_load", "permutation", "function", "leaky")


def field_scale(J, n=200, rng=None):
    """The network's own natural coupling strength: mean over ``n`` random +/-1 states s of
    mean(abs(J @ s)) -- the typical |local field| a neuron feels from J. All control-field
    strengths elsewhere in this module are expressed as fractions of this scale so that "weak"
    and "strong" mean the same thing for every J, regardless of its raw magnitude."""
    rng = np.random.default_rng() if rng is None else rng
    Jn = np.asarray(J, dtype=float)
    N = Jn.shape[0]
    samples = [float(np.mean(np.abs(Jn @ rng.choice([-1.0, 1.0], size=N)))) for _ in range(n)]
    return float(np.mean(samples))


def _nearest(state, alphabet):
    """Index of the alphabet state nearest to ``state`` (normalized Hamming distance), and that
    distance -- always returns SOME index, even if the nearest state is far away (a "leak")."""
    dists = [float(np.mean(state != a)) for a in alphabet]
    j_best = int(np.argmin(dists))
    return j_best, dists[j_best]


def induced_map(J, alphabet, u, rng=None, max_sweeps=60):
    """Apply control field ``u`` (a bias) to every alphabet state and see where it lands.

    For each alphabet state ``a``, relaxes ``HopfieldNetwork(J, bias=u)`` starting FROM ``a`` for
    up to ``max_sweeps`` sweeps, then snaps the relaxed result to the nearest alphabet state by
    Hamming distance (``_nearest`` -- always snaps to some index, whether or not the relaxed
    state actually landed on the alphabet).

    Returns ``(mapping, leak)``: ``mapping[i]`` is the index (into ``alphabet``) of the alphabet
    state nearest the relaxed result starting from ``alphabet[i]``; ``leak[i]`` is the normalized
    Hamming distance from the relaxed result to that nearest alphabet state (leak ~= 0 means it
    landed cleanly ON the alphabet; a large leak means the field pushed the state off the
    alphabet into the glassy attractor sea, not onto any of the k chosen states).
    """
    rng = np.random.default_rng() if rng is None else rng
    alphabet = [np.asarray(a, dtype=float) for a in alphabet]
    k = len(alphabet)
    net = HopfieldNetwork(J, bias=u)

    mapping = np.zeros(k, dtype=int)
    leak = np.zeros(k, dtype=float)
    for i, a in enumerate(alphabet):
        s_final = net.update_async(a, rng=rng, max_sweeps=max_sweeps)
        j_best, dist = _nearest(s_final, alphabet)
        mapping[i] = j_best
        leak[i] = dist
    return mapping, leak


def classify_map(mapping, leak, leak_tol=0.08):
    """Classify an induced map as one of ``MAP_CLASSES`` (see module docstring for what each
    class means physically). Checked in this order: LEAKY (mean leak too high -- the field
    pushed states off the alphabet, so the "map" it induces on the alphabet is not trustworthy)
    beats every other label; then IDENTITY (input did nothing); then CONSTANT_LOAD (every state
    collapses to the same one target -- storage, not computation); then, among what remains,
    PERMUTATION (a bijection -- a reversible, state-dependent GATE) or, if not bijective,
    FUNCTION (a real but irreversible state-dependent map)."""
    mapping = np.asarray(mapping, dtype=int)
    leak = np.asarray(leak, dtype=float)
    k = len(mapping)

    if float(np.mean(leak)) > leak_tol:
        return "leaky"
    if np.array_equal(mapping, np.arange(k)):
        return "identity"
    if np.all(mapping == mapping[0]):
        return "constant_load"
    if len(set(mapping.tolist())) == k:
        return "permutation"
    return "function"


def largest_cycle_length(mapping):
    """Length of the largest cycle in the functional graph of ``mapping`` (i --> mapping[i]).
    Meaningful as a "how close to one full k-cycle" measure only when ``mapping`` is bijective
    (a permutation); callers are expected to check that first."""
    mapping = np.asarray(mapping, dtype=int)
    k = len(mapping)
    seen = np.zeros(k, dtype=bool)
    best = 0
    for i in range(k):
        if seen[i]:
            continue
        length = 0
        j = i
        while not seen[j]:
            seen[j] = True
            j = int(mapping[j])
            length += 1
        best = max(best, length)
    return best


def _random_bias(field_scale_val, frac, N, rng):
    sign = rng.choice([-1.0, 1.0], size=N)
    return frac * field_scale_val * sign


def _stable_across_seeds(J, alphabet, u, expected_class, leak_tol, n_seeds, rng, max_sweeps):
    """Recompute the induced map for the SAME field ``u`` under ``n_seeds`` fresh random update
    orders / rng draws; a candidate gate only counts as a genuine, reliable gate (not a one-off
    fluke of a particular async update order) if it reproduces the same class every time."""
    for _ in range(n_seeds):
        child_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        mapping, leak = induced_map(J, alphabet, u, rng=child_rng, max_sweeps=max_sweeps)
        if classify_map(mapping, leak, leak_tol=leak_tol) != expected_class:
            return False
    return True


def gate_search(
    J,
    alphabet,
    field_scale_val,
    fracs,
    n_inputs=60,
    rng=None,
    leak_tol=0.08,
    max_sweeps=60,
    stability_seeds=3,
):
    """Sample many random control fields at several field strengths and classify each induced map.

    At each field fraction ``f`` in ``fracs``, draws ``n_inputs`` random bias fields
    ``u = f * field_scale_val * s`` (``s`` a random +/-1 sign vector) and classifies the induced
    map of each (``induced_map`` + ``classify_map``). Whenever a sample is classified
    ``permutation`` or ``function`` (a candidate state-dependent gate) and its mapping has not
    been seen before, it is re-checked at ``stability_seeds`` fresh random update orders
    (``_stable_across_seeds``) and only kept if it reproduces the same class every time -- a
    one-off fluke does not count as a discovered gate.

    Returns a dict:
      per_frac                -- list of {"frac", "counts" (dict over MAP_CLASSES), "n_inputs"}.
      distinct_permutations    -- list of distinct, stability-verified permutation mappings
                                   (tuples of ints), the SET of reversible state-dependent gates.
      permutation_fields       -- matching list of {"perm", "frac", "u"} (a representative field
                                   that induces each permutation -- for reuse by composability).
      distinct_functions       -- like ``distinct_permutations`` but for non-bijective, still
                                   state-dependent, low-leak maps.
      function_fields          -- matching representative fields.
      n_distinct_permutations  -- len(distinct_permutations).
      n_distinct_gates         -- n_distinct_permutations + len(distinct_functions).
    """
    rng = np.random.default_rng() if rng is None else rng
    J = np.asarray(J, dtype=float)
    N = J.shape[0]
    k = len(alphabet)

    per_frac = []
    perm_seen = {}
    func_seen = {}
    for frac in fracs:
        counts = {c: 0 for c in MAP_CLASSES}
        for _ in range(n_inputs):
            u = _random_bias(field_scale_val, frac, N, rng)
            mapping, leak = induced_map(J, alphabet, u, rng=rng, max_sweeps=max_sweeps)
            cls = classify_map(mapping, leak, leak_tol=leak_tol)
            counts[cls] += 1
            if cls in ("permutation", "function"):
                key = tuple(int(x) for x in mapping)
                already_kept = key in perm_seen or key in func_seen
                if not already_kept and _stable_across_seeds(
                    J, alphabet, u, cls, leak_tol, stability_seeds, rng, max_sweeps
                ):
                    rec = {"perm": key, "frac": float(frac), "u": [float(x) for x in u]}
                    (perm_seen if cls == "permutation" else func_seen)[key] = rec
        per_frac.append({"frac": float(frac), "counts": counts, "n_inputs": n_inputs})

    return {
        "per_frac": per_frac,
        "distinct_permutations": list(perm_seen.keys()),
        "permutation_fields": list(perm_seen.values()),
        "distinct_functions": list(func_seen.keys()),
        "function_fields": list(func_seen.values()),
        "n_distinct_permutations": len(perm_seen),
        "n_distinct_gates": len(perm_seen) + len(func_seen),
        "k": k,
    }


def try_increment(
    J, alphabet, field_scale_val, fracs, n_inputs=200, rng=None, leak_tol=0.08, max_sweeps=60
):
    """Search random control fields (at every fraction in ``fracs``) for one whose induced map is
    a clean k-CYCLE -- a single-cycle permutation touching all k alphabet states, i.e. an
    INCREMENT after relabeling states along the cycle.

    Only samples that ``classify_map`` labels ``"permutation"`` (bijective, low-leak, and NOT the
    trivial identity) are candidates at all -- a cycle only makes sense for a genuine,
    state-dependent permutation; the do-nothing identity map is excluded even though it is
    technically bijective. For each such candidate, ``_largest_cycle_length(mapping) / k`` is its
    "cycle fidelity": 1.0 means
    a single cycle spans the whole alphabet (a clean increment); a smaller value means the
    permutation breaks into multiple shorter cycles (a partial, non-composable increment).

    Returns a dict with ``best_cycle_fidelity`` (the max cycle fidelity found over all samples),
    ``found_clean_increment`` (whether any sample reached fidelity ~= 1.0), ``best_frac`` (the
    field fraction of the best sample), ``best_mapping`` (its mapping, as a tuple), and
    ``best_u`` (the field that induced it, for reuse e.g. by ``composability``) -- ``None`` for
    the last three if no bijective low-leak sample was found at all.
    """
    rng = np.random.default_rng() if rng is None else rng
    J = np.asarray(J, dtype=float)
    N = J.shape[0]
    k = len(alphabet)

    best_fidelity = 0.0
    best_frac = None
    best_mapping = None
    best_u = None
    found_clean = False

    for frac in fracs:
        for _ in range(n_inputs):
            u = _random_bias(field_scale_val, frac, N, rng)
            mapping, leak = induced_map(J, alphabet, u, rng=rng, max_sweeps=max_sweeps)
            if classify_map(mapping, leak, leak_tol=leak_tol) != "permutation":
                continue  # not a genuine (non-identity, bijective, low-leak) gate
            fidelity = largest_cycle_length(mapping) / k
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_frac = float(frac)
                best_mapping = tuple(int(x) for x in mapping)
                best_u = [float(x) for x in u]
            if fidelity >= 0.999:
                found_clean = True

    return {
        "best_cycle_fidelity": float(best_fidelity),
        "found_clean_increment": bool(found_clean),
        "best_frac": best_frac,
        "best_mapping": best_mapping,
        "best_u": best_u,
        "k": k,
    }


def composability(J, alphabet, u, steps=10, rng=None, max_sweeps=60):
    """Apply the SAME control field ``u`` repeatedly, snapping to the nearest alphabet state after
    every application, and ask how many chained applications stay faithful to the ideal, purely
    combinatorial iterated map.

    The "ideal" trajectory of starting state i is what repeatedly composing the FIRST-step
    induced map (``induced_map(J, alphabet, u)`` once) would predict: mapping applied to itself
    t times. The "actual" trajectory instead re-applies the PHYSICAL relaxation at every step,
    starting from wherever the previous step's snap landed -- so any leak or misclassification
    at one step can compound into a derailment at the next, exactly the failure mode a chained
    (composable) sequence of gate applications would suffer from.

    Returns ``{"steps", "fidelity_vs_step", "first_step_mapping"}`` where
    ``fidelity_vs_step[t-1]`` is the fraction of the k starting states whose ACTUAL trajectory
    still matches the IDEAL iterated map after t applications of ``u`` (t = 1..steps) -- i.e. how
    many reliable chained operations exist before the physical dynamics derails from the map.
    """
    rng = np.random.default_rng() if rng is None else rng
    J = np.asarray(J, dtype=float)
    alphabet = [np.asarray(a, dtype=float) for a in alphabet]
    k = len(alphabet)

    mapping1, _ = induced_map(J, alphabet, u, rng=rng, max_sweeps=max_sweeps)
    net = HopfieldNetwork(J, bias=u)

    actual_state = [alphabet[i].copy() for i in range(k)]
    expected_idx = list(range(k))
    fidelity_vs_step = []
    for _ in range(steps):
        new_actual_state = []
        actual_idx = []
        for i in range(k):
            s_final = net.update_async(actual_state[i], rng=rng, max_sweeps=max_sweeps)
            j_best, _ = _nearest(s_final, alphabet)
            new_actual_state.append(alphabet[j_best])
            actual_idx.append(j_best)
        actual_state = new_actual_state
        expected_idx = [int(mapping1[idx]) for idx in expected_idx]
        matches = sum(1 for a, e in zip(actual_idx, expected_idx) if a == e)
        fidelity_vs_step.append(matches / k)

    return {
        "steps": steps,
        "fidelity_vs_step": fidelity_vs_step,
        "first_step_mapping": tuple(int(x) for x in mapping1),
    }
