"""Run the offline pipeline.

Examples (PowerShell or bash, from the project root, with the venv active):

    python scripts/run_pipeline.py                     # all steps
    python scripts/run_pipeline.py --steps sentiment index
    python scripts/run_pipeline.py --nrows 50000       # quick smoke run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewrec.pipeline import STEPS, run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument("--nrows", type=int, default=None, help="read only the first N raw rows (debugging)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = run(args.steps, nrows=args.nrows)
    print({k: v for k, v in summary.items() if k.endswith("_seconds")})


if __name__ == "__main__":
    main()
