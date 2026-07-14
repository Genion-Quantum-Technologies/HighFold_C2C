"""Stateless CLI steps for Argo Workflows (ADR 0012 P1/P2).

    python -m highfold_c2c.steps generate --work-dir /work    # CPU  — C2C T5, ~1 s
    python -m highfold_c2c.steps predict  --work-dir /work    # GPU  — ColabFold / AF2
    python -m highfold_c2c.steps evaluate --work-dir /work    # CPU  — pandas

These replace `core/task_processor.py` (the polling worker). Everything the old worker did
around the science — poll the `tasks` table, claim a row, download inputs, upload outputs,
write status — is gone: Argo claims the work, the compute-foundry `fetch`/`publish` steps
move the bytes, and the operator owns the `tasks` projection. **This module never opens a
database connection or an HTTP session.**

Why the split is the whole point: Stage 1 (T5, 21 MB model, about a second) and Stage 3
(pandas) used to sit in the same long-lived pod as Stage 2, so they held the platform's
only GPU while doing nothing with it. Now only `predict` requests `nvidia.com/gpu`.

The stage boundaries are files in `/work/output`, exactly as `run_highfold_pipeline`
already laid them out — `predict.fasta` between 1 and 2, the `*_scores_*.json` files
between 2 and 3. No new contract was invented; the existing one was just made explicit.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger("highfold-step")


def _load_params(work_dir: Path) -> Dict[str, Any]:
    """Read the merged parameters the operator published at submit time.

    This replaces `task_processor.py`'s 17-key whitelist merge of `input.json` over the
    `highfold_task_params` row. That merge now happens once, in the operator, for every
    task type — which is why this image no longer needs psycopg2.
    """
    params = json.loads((work_dir / "params.json").read_text())

    # Two values are properties of the IMAGE, not of the task, so they are injected here
    # rather than carried through the database (task_processor.py:111-112 did the same).
    from highfold_c2c.config.settings import get_settings

    settings = get_settings()
    params.setdefault("checkpoint", settings.C2C_CHECKPOINT_PATH)
    params.setdefault("colabfold_bin", settings.COLABFOLD_BIN)
    return params


def stage_generate(work_dir: Path, params: Dict[str, Any]) -> None:
    """Stage 1 — C2C T5 sequence generation. Pure CPU, no GPU request."""
    from highfold_c2c.core.pipeline import run_highfold_pipeline

    if params.get("skip_generate"):
        # The caller supplied a FASTA instead of generating one. `fetch` has already put
        # it in /work/input; hand it to the pipeline, which copies it into place.
        src = work_dir / "input" / "predict.fasta"
        if not src.exists():
            raise FileNotFoundError(
                "skip_generate is set but no predict.fasta was uploaded with the job"
            )
        params = {**params, "fasta_input_path": str(src)}

    cfg = {**params, "skip_predict": True, "skip_evaluate": True}
    run_highfold_pipeline(cfg, work_dir)

    fasta = work_dir / "output" / "predict.fasta"
    if not fasta.exists():
        raise RuntimeError("stage `generate` produced no predict.fasta")
    n = sum(1 for line in fasta.read_text().splitlines() if line.startswith(">"))
    logger.info("generated %d candidate sequence(s)", n)


def stage_predict(work_dir: Path, params: Dict[str, Any]) -> None:
    """Stage 2 — ColabFold / AlphaFold2. **The only GPU stage.**

    `skip_generate=True` makes run_highfold_pipeline reuse the predict.fasta that stage 1
    left in /work/output (pipeline.py:238-249: it only copies from `fasta_input_path` when
    that key is set, otherwise it uses the FASTA already sitting there).

    Going through run_highfold_pipeline rather than shelling out to `colabfold_batch`
    directly is deliberate: the pipeline also runs `_cyclize_pdb_files()` afterwards, which
    appends the CONECT records that make the peptide cyclic. `scripts/run_predict_only.py`
    does NOT do that, so a step built on it would have silently dropped the cyclic bonds —
    the entire point of HighFold.
    """
    from highfold_c2c.core.pipeline import run_highfold_pipeline

    fasta = work_dir / "output" / "predict.fasta"
    if not fasta.exists():
        raise FileNotFoundError(f"{fasta} missing — stage `generate` did not run")

    cfg = {**params, "skip_generate": True, "skip_evaluate": True}
    cfg.pop("fasta_input_path", None)
    run_highfold_pipeline(cfg, work_dir)

    pdbs = list((work_dir / "output").glob("*.pdb"))
    if not pdbs:
        raise RuntimeError("stage `predict` produced no PDB structures")
    logger.info("predicted %d structure(s)", len(pdbs))


def stage_evaluate(work_dir: Path, params: Dict[str, Any]) -> None:
    """Stage 3 — pLDDT collection + report. Pure CPU, no GPU request.

    Implemented directly instead of via run_highfold_pipeline, because the pipeline gates
    real pLDDT collection on `not skip_predict` (pipeline.py:319-322) and stubs every score
    to NaN otherwise. In a split pipeline `predict` ran in a *different pod*, so we must
    ask for skip_predict=True — and would get a report full of NaN. This mirrors
    pipeline.py:308-344 with that one gate removed.
    """
    import numpy as np
    from c2c.evaluate import collect_plddt_scores, generate_report, read_fasta_sequences

    output_dir = work_dir / "output"
    fasta_path = output_dir / "predict.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError(f"{fasta_path} missing — stage `generate` did not run")

    seq_list = read_fasta_sequences(str(fasta_path))
    score_dict = collect_plddt_scores(str(output_dir), seq_list)

    csv_path = output_dir / "output.csv"
    generate_report(seq_list, score_dict, str(csv_path))

    valid = [v for v in score_dict.values() if not np.isnan(v)]
    if not valid:
        # Not fatal — the CSV is still produced — but it means stage 2's score files did
        # not match what the evaluator globs for, and the user gets a report of NaN.
        logger.warning(
            "no pLDDT scores matched the %d sequence(s); check the model_type used in "
            "stage `predict` (collect_plddt_scores globs for *_alphafold2_model_*)",
            len(seq_list),
        )
    else:
        logger.info(
            "evaluated %d sequence(s); pLDDT mean=%.1f min=%.1f max=%.1f",
            len(seq_list), float(np.mean(valid)), float(np.min(valid)), float(np.max(valid)),
        )


STAGES = {"generate": stage_generate, "predict": stage_predict, "evaluate": stage_evaluate}


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    p = argparse.ArgumentParser(prog="highfold_c2c.steps")
    p.add_argument("stage", choices=sorted(STAGES))
    p.add_argument("--work-dir", default="/work")
    args = p.parse_args()

    work_dir = Path(args.work_dir)
    params = _load_params(work_dir)
    logger.info("stage=%s work_dir=%s", args.stage, work_dir)

    try:
        STAGES[args.stage](work_dir, params)
    except Exception as e:
        # Argo captures this and the onExit hook lifts it into tasks.info, so it is what
        # the user finally sees. Keep it short and diagnostic.
        print(f"FATAL [{args.stage}] {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
