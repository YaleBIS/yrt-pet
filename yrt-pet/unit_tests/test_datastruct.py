#!/bin/env python
"""Integration tests for YRT-PET"""

# %% Imports

import os
import sys
import json
import tempfile
import shutil

import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as yrt

# %% Helper functions

def make_scanner():
    return {}

# %% Scanner lookup table

def test_scanner_lookup_table():
    tmp_dir = tempfile.mkdtemp()

    # Create scanner
    scanner = yrt.Scanner(scannerName='test_scanner',
                          axialFOV=25,
                          crystalSize_z=2.0,
                          crystalSize_trans=2.0,
                          crystalDepth=10.0,
                          scannerRadius=30.0,
                          detsPerRing=512,
                          numRings=8,
                          numDOI=1,
                          maxRingDiff=7,
                          minAngDiff=0,
                          detsPerBlock=32)

    det_coords_ref = yrt.DetRegular(scanner)
    det_coords_ref.generateLUT()
    lut_fname = os.path.join(tmp_dir, 'test_scanner.lut')
    det_coords_ref.writeToFile(lut_fname)
    lut_ref = np.reshape(np.fromfile(lut_fname,
                                     dtype=np.float32), [-1, 6])
    # Create mask
    mask_fname = os.path.join(tmp_dir, 'test_scanner_mask.raw')
    lut_mask = (np.arange(len(lut_ref)) % 100) != 0
    lut_mask.tofile(mask_fname)

    # Create scanner with masked detectors
    scanner_fname = os.path.join(tmp_dir, 'test_scanner.json')
    scanner_dict = {
        'VERSION': yrt.Scanner.SCANNER_FILE_VERSION,
        'scannerName' : 'test_scanner',
        'detCoord' : 'test_scanner.lut',
        'detMask' : 'test_scanner_mask.raw',
        'axialFOV': scanner.axialFOV,
        'crystalSize_z': scanner.crystalSize_z,
        'crystalSize_trans': scanner.crystalSize_trans,
        'crystalDepth': scanner.crystalDepth,
        'scannerRadius': scanner.scannerRadius,
        'fwhm': scanner.fwhm,
        'energyLLD': scanner.energyLLD,
        'collimatorRadius': scanner.collimatorRadius,
        'dets_per_ring': scanner.detsPerRing,
        'num_rings': scanner.numRings,
        'num_doi': scanner.numDOI,
        'max_ring_diff': scanner.maxRingDiff,
        'min_ang_diff': scanner.minAngDiff,
        'dets_per_block': scanner.detsPerBlock}
    with open(scanner_fname, 'wt') as fid:
        json.dump(scanner_dict, fid)

    # Load scanner
    scanner = yrt.Scanner(scanner_fname)
    lut = scanner.createLUT()
    bin_id = 10
    # Ensure scanner object is functional after extraction of lookup table
    assert (scanner.getDetectorPos(bin_id).x == lut_ref[bin_id, 0] and
            scanner.getDetectorPos(bin_id).y == lut_ref[bin_id, 1] and
            scanner.getDetectorPos(bin_id).z == lut_ref[bin_id, 2])

    assert not scanner.isLORAllowed(0, 1)
    assert scanner.isLORAllowed(1, 2)

    shutil.rmtree(tmp_dir)

# %% Image transformation


def test_image_transform():

    min_val, max_val = 1, 10
    def rescale(sample): return (max_val - min_val) * sample + min_val

    # Simple translation
    x = rescale(np.random.random([12, 13, 14])).astype(np.float32)
    img_params = yrt.ImageParams(14, 13, 12, 28.0, 26.0, 24.0, 1, 2, 3)
    img = yrt.ImageAlias(img_params)
    img.bind(x)
    v_rot = yrt.Vector3D(0.0, 0.0, 0.0)
    v_tr = yrt.Vector3D(2.0, 0.0, 0.0)
    img_t = img.transformImage(v_rot, v_tr)
    x_t = np.array(img_t, copy=False)
    np.testing.assert_allclose(x[..., :-1], x_t[..., 1:], rtol=9e-6)
    # Simple rotation
    x = rescale(np.random.random([14, 12, 12])).astype(np.float32)
    img_params = yrt.ImageParams(12, 12, 14, 26.0, 26.0, 28.0)
    img = yrt.ImageAlias(img_params)
    img.bind(x)
    v_rot = yrt.Vector3D(0.0, 0.0, np.pi / 2)
    v_tr = yrt.Vector3D(0.0, 0.0, 0.0)
    img_t = img.transformImage(v_rot, v_tr)
    x_t = np.array(img_t, copy=False)
    np.testing.assert_allclose(np.moveaxis(x, 1, 2)[..., ::-1], x_t, rtol=9e-6)
