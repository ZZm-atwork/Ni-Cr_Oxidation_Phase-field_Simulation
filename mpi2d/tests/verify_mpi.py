#!/usr/bin/env python3
"""Native MPI acceptance tests. Requires the compiled solver and a real MPI launcher.

Runs built-in tests and compares full output at several process counts. Everything
is written below a fresh timestamped directory. Standard library only.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys


def execute(command: list[str], log: Path) -> None:
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    log.write_text('$ ' + shlex.join(command) + '\n' + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def compare(reference: Path, candidate: Path) -> None:
    expected = sorted(p.name for p in reference.glob('fields_*.vtk'))
    actual = sorted(p.name for p in candidate.glob('fields_*.vtk'))
    if expected != actual:
        raise AssertionError('Different output frame sets')
    for name in expected + ['grains.csv', 'grain_history.csv', 'grain_diagnostics.csv']:
        if (reference / name).read_bytes() != (candidate / name).read_bytes():
            raise AssertionError(f'Cell/grain output differs: {candidate / name}')
    with (reference / 'diagnostics.csv').open() as fa, (candidate / 'diagnostics.csv').open() as fb:
        a, b = list(csv.DictReader(fa)), list(csv.DictReader(fb))
    if len(a) != len(b):
        raise AssertionError('Different diagnostic row counts')
    for ra, rb in zip(a, b):
        for k in ra:
            x, y = float(ra[k]), float(rb[k])
            if abs(x-y) > 1e-10 * (1+abs(x)):
                raise AssertionError(f'Diagnostic mismatch: step={ra["step"]}, column={k}')


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', default='./nicr2d_mpi')
    p.add_argument('--launcher', default='mpirun')
    p.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4])
    p.add_argument('--output', default=None)
    a = p.parse_args()
    if not a.ranks or min(a.ranks) < 1:
        p.error('Process counts must be positive')
    binary = str(Path(a.binary).resolve())
    if not Path(binary).is_file():
        p.error('Compile the MPI binary first: make')
    out = Path(a.output or ('mpi_verification_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(exist_ok=False, parents=True)
    cases = {
        'mu': ['--bc', 'mu', '--ck', 'off', '--steps', '1000'],
        'anchored': ['--bc', 'mu', '--ck', 'off', '--left-edge-sites', '2', '--steps', '1000'],
        'center': ['--bc', 'closed', '--ck', 'off', '--nucleation', 'center', '--steps', '500'],
        'O_off': ['--bc', 'closed', '--ck', 'off', '--nucleation', 'center', '--oxygen-diffusion', 'off', '--steps', '500'],
        'legacy_Ck': ['--bc', 'closed', '--ck', 'legacy', '--nucleation', 'center', '--steps', '500'],
    }
    rank_set = sorted(set([1] + a.ranks))
    for n in rank_set:
        prefix = shlex.split(a.launcher) + ['-np', str(n), binary]
        execute(prefix + ['--self-test'], out / f'selftest_np{n}.log')
        for name, args in cases.items():
            folder = out / f'{name}_np{n}'
            common = ['--nx', '61', '--ny', '31', '--grain-diameter-grid', '30',
                      '--seed', '2001', '--grain-evolution', 'on', '--eta-output', 'all',
                      '--bounds', 'legacy', '--dt', '1e-12', '--output-every', '250']
            execute(prefix + common + args + ['--out', str(folder)], out / f'{name}_np{n}.log')
            if n != 1:
                compare(out / f'{name}_np1', folder)
            print(f'PASS {name}, np={n}')
    (out / 'PASS.txt').write_text('Native MPI built-in and output-regression tests passed.\n'
                                 'These short tests do not establish production stability or parameter calibration.\n')
    print(f'Native MPI checks complete: {out}')
    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, RuntimeError, AssertionError, subprocess.TimeoutExpired) as exc:
        print(f'VERIFICATION FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
