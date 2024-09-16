#!/bin/env python
"""Integration tests for YRT-PET"""

# %% Imports

import os
import sys

import numpy as np

fold_py = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(fold_py))
import pyyrtpet as gc

import helper as _helper

dataset_paths = _helper.dataset_paths
out_paths = _helper.out_paths
ref_paths = _helper.ref_paths
util_paths = _helper.util_paths
fold_data = _helper.fold_data
fold_out = _helper.fold_out
fold_bin = _helper.fold_bin

# %% Scanner lookup table


def test_scanner_lookup_table():
    scanner = gc.GCScannerOwned(
        util_paths['Geometry_2panels_large_3x3x20mm_rot_gc_json'])
    lut = scanner.createLUT()
    bin_id = 10
    # Ensure scanner object is functional after extraction of lookup table
    assert (scanner.getDetectorPos(bin_id).x == lut[bin_id, 0] and
            scanner.getDetectorPos(bin_id).y == lut[bin_id, 1] and
            scanner.getDetectorPos(bin_id).z == lut[bin_id, 2])

# %% Image transformation


def test_image_transform():
    # Simple translation
    x = np.random.random([12, 13, 14])
    img_params = gc.ImageParams(14, 13, 12, 28.0, 26.0, 24.0)
    img = gc.ImageAlias(img_params)
    img.Bind(x)
    v_rot = gc.GCVector(0.0, 0.0, 0.0)
    v_tr = gc.GCVector(2.0, 0.0, 0.0)
    img_t = img.transformImage(v_rot, v_tr)
    x_t = np.array(img_t, copy=False)
    np.testing.assert_allclose(x[..., :-1], x_t[..., 1:])
    # Simple rotation
    x = np.random.random([14, 12, 12])
    img_params = gc.ImageParams(12, 12, 14, 26.0, 26.0, 28.0)
    img = gc.ImageAlias(img_params)
    img.Bind(x)
    v_rot = gc.GCVector(0.0, 0.0, np.pi / 2)
    v_tr = gc.GCVector(0.0, 0.0, 0.0)
    img_t = img.transformImage(v_rot, v_tr)
    x_t = np.array(img_t, copy=False)
    np.testing.assert_allclose(np.moveaxis(x, 1, 2)[..., ::-1], x_t)
