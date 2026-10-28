"""Master command for all Engineered Science confirmatory experiments."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .q1_common import finalize_manifest, parser, resolved_config, write_json


GROUPS = ("attractor", "reservoir", "multivalued", "rbm")


def _run_group(group, config, out, resume, force):
    if group == "attractor":
        from .run_phase2_attractor_recall_q1 import run
    elif group == "reservoir":
        from .run_phase3_reservoir_q1 import run
    elif group == "multivalued":
        from .run_phase4_multivalued_q1 import run
    else:
        from .run_phase5_rbm_q1 import run
    return run(config, Path(out), resume=resume, force=force)


def main(argv=None):
    p = parser(__doc__)
    p.add_argument(
        "--groups",
        default=",".join(GROUPS),
        help="Comma-separated subset: attractor,reservoir,multivalued,rbm",
    )
    args = p.parse_args(argv)
    config = resolved_config(args)
    if args.force:
        args.resume = False
    selected = [g.strip() for g in args.groups.split(",") if g.strip()]
    unknown = sorted(set(selected) - set(GROUPS))
    if unknown:
        p.error(f"unknown groups: {unknown}")
    if not selected:
        p.error("select at least one experiment group")
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / "config_used.json", config)
    required: list[Path] = [args.out / "config_used.json"]
    workers = min(max(1, args.jobs), len(selected))
    if workers == 1:
        for group in selected:
            required.extend(_run_group(group, config, args.out, args.resume, args.force))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_run_group, group, config, args.out, args.resume, args.force)
                for group in selected
            ]
            for future in futures:
                required.extend(future.result())
    finalize_manifest(args.out, config, required)
    print(f"Q1 upgrade complete: {args.out}")


if __name__ == "__main__":
    main()
