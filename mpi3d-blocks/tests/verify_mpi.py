#!/usr/bin/env python3
"""Verify 3D Cartesian solver fields, diagnostics, and ownership across MPI layouts.

Uses Python's standard library only. Default execution uses a native MPI launcher.
--emulated explicitly selects the separate thread-based test harness, NOT native MPI.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys


def execute(command: list[str], log: Path, timeout: int) -> None:
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    log.write_text('$ ' + shlex.join(command) + '\n' + result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def compare(reference: Path, candidate: Path) -> dict[str, str]:
    expected = sorted(p.name for p in reference.glob('fields_*.vtk'))
    actual = sorted(p.name for p in candidate.glob('fields_*.vtk'))
    if not expected or expected != actual:
        raise AssertionError('Different or missing output frame sets')
    optional = [name for name in ('grains.csv', 'grain_history.csv', 'grain_diagnostics.csv')
                if (reference / name).exists()]
    hashes: dict[str, str] = {}
    for name in expected + optional:
        a, b = (reference / name).read_bytes(), (candidate / name).read_bytes()
        if a != b:
            raise AssertionError(f'Cell/grain output differs: {candidate / name}')
        hashes[name] = hashlib.sha256(b).hexdigest()
    with (reference / 'diagnostics.csv').open() as fa, (candidate / 'diagnostics.csv').open() as fb:
        a, b = list(csv.DictReader(fa)), list(csv.DictReader(fb))
    if len(a) != len(b) or not a:
        raise AssertionError('Different or missing diagnostic row counts')
    for ra, rb in zip(a, b):
        if set(ra) != set(rb):
            raise AssertionError('Different diagnostic columns')
        for key in ra:
            x, y = float(ra[key]), float(rb[key])
            if not math.isfinite(x) or not math.isfinite(y) or abs(x-y) > 1e-10 * (1+abs(x)):
                raise AssertionError(f'Diagnostic mismatch: step={ra["step"]}, column={key}')
    return hashes


def audit_layout(folder: Path, ranks: int, fixed: list[str]) -> tuple[int, int, int]:
    parameters = dict(line.split('=', 1) for line in (folder/'parameters.txt').read_text().splitlines() if '=' in line)
    if parameters['mpi_decomposition'] != '3D_Cartesian_blocks':
        raise AssertionError('Output does not identify Cartesian block decomposition')
    dims = tuple(int(parameters['mpi_p'+a]) for a in 'xyz')
    mesh = tuple(int(parameters['n'+a]) for a in 'xyz')
    if math.prod(dims) != ranks or int(parameters['mpi_processes']) != ranks:
        raise AssertionError('Incorrect process-grid product in metadata')
    for flag, value in zip(fixed[::2], fixed[1::2]):
        if int(value) and dims['xyz'.index(flag[-1])] != int(value):
            raise AssertionError('A fixed process-grid axis was changed')
    with (folder/'mpi_layout.csv').open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != ranks:
        raise AssertionError('Wrong number of block metadata rows')
    seen: set[tuple[int, int, int]] = set()
    coords_seen: set[tuple[int, int, int]] = set()
    for rank, row in enumerate(rows):
        if int(row['rank']) != rank:
            raise AssertionError('Invalid rank ordering')
        lo, hi = [], []
        coords = tuple(int(row['coord_'+a]) for a in 'xyz')
        coords_seen.add(coords)
        for axis, a in enumerate('xyz'):
            q = coords[axis]
            base, extra = divmod(mesh[axis], dims[axis])
            count = base + (q < extra)
            begin = 1 + q*base + min(q, extra)
            lo.append(int(row['ijk'[axis]+'_first']))
            hi.append(int(row['ijk'[axis]+'_last']))
            if not (0 <= q < dims[axis]) or lo[-1] != begin or hi[-1] != begin+count-1 or int(row['local_n'+a]) != count:
                raise AssertionError('Incorrect uneven-block bounds')
        cells = {(i,j,k) for i in range(lo[0],hi[0]+1)
                 for j in range(lo[1],hi[1]+1) for k in range(lo[2],hi[2]+1)}
        if len(cells) != int(row['owned_cells']) or seen.intersection(cells):
            raise AssertionError('Overlapping blocks or incorrect owned-cell count')
        seen.update(cells)
    if len(seen) != math.prod(mesh) or len(coords_seen) != ranks:
        raise AssertionError('Incomplete mesh/process-coordinate coverage')
    return dims


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', default='./nicr3d_mpi')
    p.add_argument('--launcher', default='mpirun', help='Launcher and flags, e.g. "mpirun --oversubscribe"')
    p.add_argument('--ranks', type=int, nargs='+', default=[1, 2, 4, 8])
    p.add_argument('--output', default=None)
    p.add_argument('--timeout', type=int, default=240, help='Per-invocation timeout in seconds')
    p.add_argument('--emulated', action='store_true', help='Use the thread harness, NOT native MPI')
    p.add_argument('--skip-self-tests', action='store_true', help='Run output/ownership regression only')
    p.add_argument('--reference-binary', help='Optional prior solver binary, run directly in single-process mode')
    p.add_argument('--auto-only', action='store_true', help='Skip additional manual-layout comparisons')
    a = p.parse_args()
    if min(a.ranks) < 1 or max(a.ranks) > 64 or a.timeout < 1:
        p.error('Tests support 1..64 ranks and a positive timeout')
    binary = str(Path(a.binary).resolve())
    if not Path(binary).is_file():
        p.error('Compile the binary first')
    reference_binary = str(Path(a.reference_binary).resolve()) if a.reference_binary else None
    if reference_binary and not Path(reference_binary).is_file():
        p.error('Reference binary not found')
    out = Path(a.output or ('mpi_verification_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    out.mkdir(exist_ok=False, parents=True)
    backend = 'thread emulation (NOT native MPI)' if a.emulated else 'native MPI'
    cases = {
        'fixed_mu': ['--grain-evolution', 'off', '--bc', 'mu', '--nucleation', 'off'],
        'eta_mu': ['--grain-evolution', 'on', '--bc', 'mu', '--nucleation', 'off'],
        'anchored': ['--grain-evolution', 'on', '--bc', 'mu', '--left-face-sites', '2', '--nucleation', 'off'],
        'center': ['--grain-evolution', 'on', '--bc', 'closed', '--nucleation', 'center'],
        'fixed_center': ['--grain-evolution', 'off', '--bc', 'closed', '--nucleation', 'center'],
        'O_off': ['--grain-evolution', 'on', '--bc', 'closed', '--nucleation', 'center', '--oxygen-diffusion', 'off'],
        'legacy_Ck': ['--grain-evolution', 'on', '--bc', 'closed', '--nucleation', 'center', '--ck', 'legacy'],
        'eta_frozen': ['--grain-evolution', 'on', '--eta-L', '0', '--bc', 'mu', '--nucleation', 'off'],
        'eta_concentration': ['--grain-evolution', 'on', '--bc', 'concentration', '--nucleation', 'off'],
        'single_crystal': ['--grains', 'off', '--bc', 'concentration', '--nucleation', 'off'],
    }
    common = ['--nx','31','--ny','17','--nz','15','--grain-diameter-grid','18',
              '--radius-grid','3.5','--seed','2001','--eta-output','all',
              '--ck','off','--bounds','legacy','--dt','1e-12','--steps','30','--output-every','30']
    layouts: list[tuple[str, int, list[str]]] = [('auto_np'+str(n),n,[]) for n in sorted(set([1]+a.ranks))]
    if not a.auto_only:
        for n in sorted(set(a.ranks)):
            if 1 < n <= 31:
                layouts.append(('xslab_np'+str(n),n,['--px',str(n),'--py','1','--pz','1']))
        if 4 in a.ranks:
            layouts.append(('yz_np4',4,['--px','1','--py','2','--pz','2']))
        if 8 in a.ranks:
            layouts.append(('yz_np8',8,['--px','1','--py','2','--pz','4']))
            layouts.append(('partial_np8',8,['--py','4']))
        if 16 in a.ranks:
            layouts.append(('zbrick_np16',16,['--px','2','--py','2','--pz','4']))
    summary: list[dict] = []
    for label, n, fixed in layouts:
        prefix = [binary,str(n)] if a.emulated else shlex.split(a.launcher)+['-np',str(n),binary]
        if not a.skip_self_tests:
            execute(prefix+fixed+['--self-test'],out/(label+'_selftest.log'),a.timeout)
        for name, args in cases.items():
            folder = out/(name+'_'+label)
            execute(prefix+common+args+fixed+['--out',str(folder)],out/(name+'_'+label+'.log'),a.timeout)
            dims = audit_layout(folder,n,fixed)
            ref = out/(name+'_auto_np1')
            hashes = compare(ref,folder)
            if n==1 and reference_binary:
                old=out/(name+'_prior_single_rank')
                execute([reference_binary]+common+args+['--out',str(old)],out/(name+'_prior_single_rank.log'),a.timeout)
                compare(old,folder)
            summary.append({'case':name,'layout':label,'ranks':n,'grid':dims,'backend':backend,
                            'exact_fields_and_grain_outputs':True,'diagnostics_within_tolerance':True,
                            'ownership_audit_passed':True,'sha256':hashes})
            (out/'summary.json').write_text(json.dumps(summary,indent=2))
            print(f'PASS {name}, {label}, grid={dims}, {backend}',flush=True)
    (out/'PASS.txt').write_text(f'Backend: {backend}\n{len(summary)} output/ownership cases passed.\n'
        f'Prior single-rank comparison: {bool(reference_binary)}\n'
        'Byte-exact VTK and grain outputs; mass diagnostics checked with 1e-10*(1+abs(reference)).\n'
        'These short tests do not establish production stability, MPI performance, or material calibration.\n')
    print(f'Checks complete: {out}')
    return 0

if __name__=='__main__':
    try:
        sys.exit(main())
    except (OSError,ValueError,RuntimeError,AssertionError,subprocess.TimeoutExpired) as exc:
        print(f'VERIFICATION FAILED: {exc}',file=sys.stderr)
        sys.exit(1)
