#!/usr/bin/env python3
"""Compare the optimized solver with the included exact pre-optimization source.
Python standard library only. Optional --with-vtk uses an installed VTK reader.
Native MPI is tested only with --backend native in an MPI-capable allocation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['phi', 'conc_O', 'conc_Cr', 'mu_O', 'mu_Cr', 'grain_sum', 'GB_mask']

def run(args):
    p = subprocess.run([str(x) for x in args], cwd=ROOT, capture_output=True, text=True, timeout=180)
    if p.returncode:
        raise RuntimeError(f"Command failed: {shlex.join([str(x) for x in args])}\n{p.stdout}\n{p.stderr}")
    return p.stdout

def decode_vti(path):
    content = path.read_bytes()
    prefix, data = content.split(b'<AppendedData encoding="raw">\n_', 1)
    xml = ET.fromstring(prefix + b'</VTKFile>')
    image = xml.find('ImageData')
    arrays = {}
    for a in image.find('Piece').find('PointData'):
        offset = int(a.attrib['offset'])
        n, block_size, last = struct.unpack_from('<QQQ', data, offset)
        sizes = struct.unpack_from('<' + 'Q' * n, data, offset + 24)
        cursor = offset + 24 + 8 * n
        blocks = []
        for i, size in enumerate(sizes):
            raw = zlib.decompress(data[cursor:cursor + size]); cursor += size
            expected = last if i == n - 1 and last else block_size
            assert len(raw) == expected
            blocks.append(raw)
        arrays[a.attrib['Name']] = (a.attrib['type'], b''.join(blocks))
    assert list(arrays) == NAMES
    return image.attrib, arrays

def ascii_arrays(path):
    text = path.read_text()
    return {name: [float(v) for v in payload.split()]
            for name, payload in re.findall(r'SCALARS (\S+) double 1\nLOOKUP_TABLE default\n(.*?)(?=SCALARS|\Z)', text, re.S)}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['serial', 'emulated', 'native'], default='serial')
    p.add_argument('--ranks', nargs='+', type=int, default=[4])
    p.add_argument('--cxx', default=None)
    p.add_argument('--launcher', default=os.environ.get('MPIEXEC', 'mpirun'))
    p.add_argument('--quick', action='store_true')
    p.add_argument('--with-vtk', action='store_true')
    args = p.parse_args()
    if any(r < 1 for r in args.ranks): p.error('ranks must be positive')
    if args.backend == 'emulated' and any(r > 32 for r in args.ranks): p.error('thread emulator supports at most 32 ranks')
    cxx = shlex.split(args.cxx or ('mpic++' if args.backend == 'native' else 'g++'))
    cases = [0, 12, 24, 36, 47] if args.quick else list(range(48))
    result = {'backend': args.backend, 'ranks': args.ranks if args.backend != 'serial' else [1],
              'cases': cases, 'reference_sha256': hashlib.sha256((ROOT/'tests/reference/NiCr3D_grains.cpp').read_bytes()).hexdigest(),
              'state_checks': 0, 'ascii_checks': 0, 'vti_checks': 0, 'passed': False}
    with tempfile.TemporaryDirectory(prefix='nicr-verify-') as tmp:
        work = Path(tmp)
        common = cxx + ['-O2', '-std=c++17', '-I.', '-DNICR_SERIAL_VERIFY']
        ref = work/'reference'; opt = work/'optimized'
        run(common + ['-DNICR_REFERENCE', 'tests/regression_driver.cpp', '-o', ref, '-lz'])
        run(common + ['tests/regression_driver.cpp', '-o', opt, '-lz'])
        parallel = work/'parallel'
        if args.backend != 'serial':
            flags = ['-DNICR_MPI_THREAD_TEST', '-pthread'] if args.backend == 'emulated' else []
            run(cxx + ['-O2', '-std=c++17', '-I.'] + flags + ['tests/regression_driver.cpp', '-o', parallel, '-lz'])
        for case in cases:
            reference = work/f'{case}_ref'
            run([ref, case, reference])
            targets = [('serial', [opt]), ('uncached', [opt])]
            if args.backend != 'serial':
                for rank in args.ranks:
                    command = [parallel, rank] if args.backend == 'emulated' else shlex.split(args.launcher) + ['-np', str(rank), str(parallel)]
                    targets.append((f'{args.backend}_{rank}', command))
            expected = ascii_arrays(reference/'fields.vtk')
            for label, command in targets:
                out = work/f'{case}_{label}'
                run(command + [str(case), str(out)] + (['uncached'] if label == 'uncached' else []))
                for file in ['initial.bin', 'final.bin']:
                    assert (out/file).read_bytes() == (reference/file).read_bytes(), (case, label, file)
                    result['state_checks'] += 1
                assert (out/'fields.vtk').read_bytes() == (reference/'fields.vtk').read_bytes(), (case,label,'VTK')
                result['ascii_checks'] += 1
                if label in ['serial', 'uncached']:
                    for file in ['diagnostics.csv', 'grain_diagnostics.csv', 'grain_history.csv']:
                        assert (out/file).read_bytes() == (reference/file).read_bytes(), (case,label,file)
                geometry, arrays = decode_vti(out/'fields.vti')
                assert geometry['WholeExtent'] == '0 24 0 14 0 12'
                for name, (kind, raw) in arrays.items():
                    fmt = 'f' if kind == 'Float32' else 'd'
                    expected_bytes = struct.pack('<' + fmt * len(expected[name]), *expected[name])
                    assert raw == expected_bytes, (case,label,name)
                if args.with_vtk:
                    from vtkmodules.vtkIOXML import vtkXMLImageDataReader
                    reader = vtkXMLImageDataReader();reader.SetFileName(str(out/'fields.vti'));reader.Update()
                    assert reader.GetErrorCode() == 0 and reader.GetOutput().GetNumberOfPoints() == 25*15*13
                result['vti_checks'] += 1
            print(f'PASS case {case}', flush=True)
    result['passed'] = True
    dest = ROOT/'tests/logs'/f'verify_{args.backend}.json'
    dest.parent.mkdir(parents=True, exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
