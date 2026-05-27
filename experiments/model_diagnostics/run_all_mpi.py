"""MPI driver: run one generator diagnostic per rank.

Each rank shells out to run_diagnostic.py for its assigned generator(s).
Generator keys are read from config.GENERATORS (preserves dict order).

Launch via SLURM: srun python run_all_mpi.py [--n_realizations N] [--n_years Y]
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from mpi4py import MPI

from config import GENERATORS

HERE = Path(__file__).resolve().parent


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_realizations", "-r", type=int, default=3)
    parser.add_argument("--n_years", "-y", type=int, default=30)
    parser.add_argument("--seed", "-s", type=int, default=42)
    args = parser.parse_args()

    gen_keys = list(GENERATORS.keys())

    if rank == 0:
        print(f"[driver] {size} MPI ranks, {len(gen_keys)} generators", flush=True)
        if size != len(gen_keys):
            print(
                f"[driver] WARNING: ntasks ({size}) != #generators ({len(gen_keys)}); "
                f"using round-robin assignment",
                flush=True,
            )

    my_keys = gen_keys[rank::size]
    results = []

    for key in my_keys:
        print(f"[rank {rank}] START {key}", flush=True)
        t0 = time.time()
        proc = subprocess.run(
            [
                sys.executable,
                "run_diagnostic.py",
                "--generator", key,
                "--n_realizations", str(args.n_realizations),
                "--n_years", str(args.n_years),
                "--seed", str(args.seed),
            ],
            cwd=HERE,
            check=False,
        )
        elapsed = time.time() - t0
        status = "PASS" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"[rank {rank}] DONE  {key} {status} {elapsed:.1f}s", flush=True)
        results.append((key, proc.returncode, elapsed))

    comm.Barrier()
    all_results = comm.gather(results, root=0)

    if rank == 0:
        flat = [r for rank_results in all_results for r in rank_results]
        print("\n" + "=" * 60)
        print("  Summary")
        print("=" * 60)
        n_pass = 0
        for key, rc, elapsed in flat:
            status = "PASS" if rc == 0 else f"FAIL(rc={rc})"
            print(f"  {key:35s} {status:12s} {elapsed:7.1f}s")
            if rc == 0:
                n_pass += 1
        print("=" * 60)
        print(f"  {n_pass}/{len(flat)} generators succeeded")
        print("=" * 60)
        sys.exit(0 if n_pass == len(flat) else 1)


if __name__ == "__main__":
    main()
