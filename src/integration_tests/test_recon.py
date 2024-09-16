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

# %% Tests

def test_mlem_simple():
    img_params = gc.GCImageParams(util_paths['img_params_500'])
    scanner = gc.GCScannerOwned(util_paths['SAVANT_json'])
    dataset = gc.GCListModeLUTOwned(scanner, dataset_paths['test_mlem_simple'])
    sens_img = gc.GCImageOwned(img_params, util_paths['SensImageSAVANT500'])
    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths['test_mlem_simple'], ref_paths['test_mlem_simple'],
        num_MLEM_iterations=30)


def _test_mlem_helper(dset):
    """Helper function for motion MLEM tests"""
    test_name = 'test_mlem_{}'.format(dset)
    img_params = gc.GCImageParams(util_paths['img_params_500'])
    scanner = gc.GCScannerOwned(util_paths['SAVANT_json'])
    dataset = gc.GCListModeLUTOwned(scanner, dataset_paths[test_name][0])
    sens_img = gc.GCImageOwned(img_params, util_paths['SensImageSAVANT500'])

    warper = gc.ImageWarperMatrix()
    warper.setImageHyperParam(img_params)
    warper.setFramesParamFromFile(dataset_paths[test_name][1])
    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths[test_name], ref_paths[test_name],
        warper=warper, num_MLEM_iterations=10, hard_threshold=100.0)


def test_mlem_piston():
    _test_mlem_helper('piston')


def test_mlem_wobble():
    _test_mlem_helper('wobble')


def test_mlem_yesMan():
    _test_mlem_helper('yesMan')


def test_bwd():
    img_params = gc.GCImageParams(util_paths['img_params_500'])
    scanner = gc.GCScannerOwned(util_paths['SAVANT_json'])
    dataset = gc.GCListModeLUTOwned(scanner, dataset_paths['test_bwd'])

    out_img = gc.GCImageOwned(img_params)
    out_img.allocate()
    out_img.setValue(0.0)

    gc.backProject(scanner, out_img, dataset)
    out_img.applyThreshold(out_img, 1, 0, 0, 1, 0)

    out_img.writeToFile(out_paths['test_bwd'])

    ref_img = gc.GCImageOwned(img_params, ref_paths['test_bwd'])
    rmse = _helper.get_rmse(np.array(out_img, copy=False),
                            np.array(ref_img, copy=False))
    assert rmse < 10**-4


def test_sens():
    img_params = gc.GCImageParams(util_paths['img_params_500'])
    scanner = gc.GCScannerOwned(util_paths['SAVANT_json'])
    dataset = gc.GCUniformHistogram(scanner)

    osem = gc.createOSEM(scanner)
    osem.imageParams = img_params
    osem.setDataInput(dataset)
    out_imgs = osem.generateSensitivityImages()

    out_img = out_imgs[0]
    out_img.writeToFile(out_paths['test_sens'])

    ref_img = gc.GCImageOwned(img_params, ref_paths['test_sens'])
    rmse = _helper.get_rmse(np.array(out_img, copy=False),
                            np.array(ref_img, copy=False))
    assert rmse < 10**-4


def _test_savant_motion_post_mc(test_name: str):
    img_params = gc.GCImageParams(util_paths['img_params_500'])
    file_list = dataset_paths[test_name]
    image_list = []
    for fname in file_list[:-1]:
        image_list.append(gc.GCImageOwned(img_params, fname))

    warper = gc.ImageWarperFunction()
    warper.setImageHyperParam([img_params.nx, img_params.ny, img_params.nz],
                              [img_params.length_x, img_params.length_y,
                               img_params.length_z])
    warper.setFramesParamFromFile(dataset_paths[test_name][len(file_list) - 1])

    out_img = gc.GCImageOwned(img_params)
    out_img.allocate()
    out_img.setValue(0.0)
    for i, image in enumerate(image_list):
        warper.warpImageToRefFrame(image, i)
        image.addFirstImageToSecond(out_img)

    out_img.writeToFile(out_paths[test_name])

    ref_img = gc.GCImageOwned(img_params, ref_paths[test_name])
    rmse = _helper.get_rmse(np.array(out_img, copy=False),
                            np.array(ref_img, copy=False))
    assert rmse < 10**-4


def test_post_recon_mc_piston():
    _test_savant_motion_post_mc('test_post_recon_mc_piston')


def test_post_recon_mc_wobble():
    _test_savant_motion_post_mc('test_post_recon_mc_wobble')


def test_psf():
    img_params = gc.GCImageParams(50, 50, 25, 50, 50, 25, 0.0, 0.0, 0.0)
    image_in = gc.GCImageOwned(img_params, dataset_paths['test_psf'][0])
    oper_psf = gc.GCOperatorPsf(img_params, dataset_paths['test_psf'][1])
    image_ref = gc.GCImageOwned(img_params, ref_paths['test_psf'])
    image_out = gc.GCImageOwned(img_params)
    image_out.allocate()
    image_out.setValue(0.0)

    oper_psf.applyA(image_in, image_out)

    image_out.writeToFile(out_paths['test_psf'])
    rmse = _helper.get_rmse(np.array(image_out, copy=False),
                            np.array(image_ref, copy=False))
    assert rmse < 10**-4


def test_psf_adjoint():
    rng = np.random.default_rng(13)

    nx = rng.integers(1, 30)
    ny = rng.integers(1, 30)
    nz = rng.integers(1, 20)
    sx = rng.random() * 5 + 0.01
    sy = rng.random() * 10 + 0.01
    sz = rng.random() * 10 + 0.01
    ox = 0.0
    oy = 0.0
    oz = 0.0
    img_params = gc.GCImageParams(nx, ny, nz, sx, sy, sz, ox, oy, oz)

    img_X = gc.GCImageAlias(img_params)
    img_Y = gc.GCImageAlias(img_params)

    img_X_a = rng.random([nz, ny, nx]) * 10 - 5
    img_Y_a = rng.random([nz, ny, nx]) * 10 - 5
    img_X.Bind(img_X_a)
    img_Y.Bind(img_Y_a)

    oper_psf = gc.GCOperatorPsf(img_params, dataset_paths['test_psf'][1])

    Ax = gc.GCImageOwned(img_params)
    Aty = gc.GCImageOwned(img_params)
    Ax.allocate()
    Ax.setValue(0.0)
    Aty.allocate()
    Aty.setValue(0.0)

    oper_psf.applyA(img_X, Ax)
    oper_psf.applyAH(img_Y, Aty)

    dot_Ax_y = Ax.dot_product(img_Y)
    dot_x_Aty = img_X.dot_product(Aty)

    assert abs(dot_Ax_y - dot_x_Aty) < 10**-4


def test_flat_panel_mlem_tof():
    img_params = gc.GCImageParams(util_paths['img_params_3.0'])
    scanner = gc.GCScannerOwned(
        util_paths['Geometry_2panels_large_3x3x20mm_rot_gc_json'])
    dataset = gc.GCListModeLUTDOIOwned(
        scanner, dataset_paths['test_flat_panel_mlem_tof'][0], True)
    sens_img = gc.GCImageOwned(img_params,
                               dataset_paths['test_flat_panel_mlem_tof'][1])

    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths['test_flat_panel_mlem_tof'],
        ref_paths['test_flat_panel_mlem_tof'][0],
        num_MLEM_iterations=5, num_OSEM_subsets=12, num_threads=20,
        tof_width_ps=70, tof_n_std=5)


def test_flat_panel_mlem_tof_exec():
    exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    exec_str += ' --scanner ' + \
        util_paths['Geometry_2panels_large_3x3x20mm_rot_gc_json']
    exec_str += ' --params ' + util_paths['img_params_3.0']
    exec_str += ' --input ' + dataset_paths['test_flat_panel_mlem_tof'][0]
    exec_str += ' --format LM-DOI --projector DD_GPU'
    exec_str += ' --sens ' + dataset_paths['test_flat_panel_mlem_tof'][2]
    exec_str += ' --flag_tof --tof_width_ps 70 --tof_n_std 5'
    exec_str += ' --num_iterations 10 --num_threads 20'
    exec_str += ' --out ' + out_paths['test_flat_panel_mlem_tof_exec']
    ret = os.system(exec_str)
    assert ret == 0

    img_params = gc.GCImageParams(util_paths['img_params_3.0'])
    ref_img = gc.GCImageOwned(img_params,
                              ref_paths['test_flat_panel_mlem_tof'][1])
    out_img = gc.GCImageOwned(img_params,
                              out_paths['test_flat_panel_mlem_tof_exec'])
    np.testing.assert_allclose(np.array(ref_img, copy=False),
                               np.array(out_img, copy=False),
                               atol=1e-5)


def test_subsets_savant_siddon():
    scanner = gc.GCScannerOwned(util_paths["SAVANT_json"])
    img_params = gc.GCImageParams(util_paths["img_params_500"])
    lm = gc.GCListModeLUTOwned(scanner, dataset_paths["test_subsets_savant"])
    _helper._test_subsets(scanner, img_params, lm, projector='Siddon')


def test_subsets_savant_dd():
    scanner = gc.GCScannerOwned(util_paths["SAVANT_json"])
    img_params = gc.GCImageParams(util_paths["img_params_500"])
    lm = gc.GCListModeLUTOwned(scanner, dataset_paths["test_subsets_savant"])
    _helper._test_subsets(scanner, img_params, lm, projector='DD')


def test_adjoint_uhr2d_siddon():
    scanner = gc.GCScannerOwned(util_paths["UHR2D_json"])
    img_params = gc.GCImageParams(util_paths["img_params_2d"])
    his = gc.GCListModeLUTOwned(scanner, dataset_paths["test_adjoint_uhr2d"])
    _helper._test_adjoint(scanner, img_params, his, projector='Siddon')


def test_adjoint_uhr2d_multi_ray_siddon():
    scanner = gc.GCScannerOwned(util_paths["UHR2D_json"])
    img_params = gc.GCImageParams(util_paths["img_params_2d"])
    his = gc.GCListModeLUTOwned(scanner, dataset_paths["test_adjoint_uhr2d"])
    _helper._test_adjoint(scanner, img_params, his, projector='Siddon',
                          num_rays=4)


def test_adjoint_uhr2d_dd():
    scanner = gc.GCScannerOwned(util_paths["UHR2D_json"])
    img_params = gc.GCImageParams(util_paths["img_params_2d"])
    his = gc.GCListModeLUTOwned(scanner, dataset_paths["test_adjoint_uhr2d"])
    _helper._test_adjoint(scanner, img_params, his, projector='DD')


def test_osem_his_2d():
    recon_exec_str = os.path.join(fold_bin, 'yrtpet_reconstruct')
    recon_exec_str += " --scanner " + util_paths['UHR2D_json']
    recon_exec_str += " --params " + util_paths["img_params_2d"]
    recon_exec_str += " --out " + out_paths['test_osem_his_2d'][0]
    recon_exec_str += " --out_sens " + out_paths['test_osem_his_2d'][1]
    recon_exec_str += " --input " + dataset_paths['test_osem_his_2d']
    recon_exec_str += " --format H"
    recon_exec_str += " --num_subsets 5"
    recon_exec_str += " --num_iterations 100"
    print("Running: " + recon_exec_str)
    os.system(recon_exec_str)

    img_params = gc.GCImageParams(util_paths['img_params_2d'])
    for i in range(5):
        ref_gensensimg = gc.GCImageOwned(img_params,
                                         ref_paths['test_osem_his_2d'][1][i])
        out_gensensimg = gc.GCImageOwned(img_params,
                                         out_paths['test_osem_his_2d'][2][i])
        rmse = _helper.get_rmse(np.array(ref_gensensimg, copy=False),
                                np.array(out_gensensimg, copy=False))
        assert rmse < 10**-4

    ref_gensensimg = gc.GCImageOwned(img_params,
                                     ref_paths['test_osem_his_2d'][0])
    out_gensensimg = gc.GCImageOwned(img_params,
                                     out_paths['test_osem_his_2d'][0])
    rmse = _helper.get_rmse(np.array(ref_gensensimg, copy=False),
                            np.array(out_gensensimg, copy=False))
    assert rmse < 10**-4


def test_osem_siddon_multi_ray():
    num_siddon_rays = 6
    img_params = gc.GCImageParams(util_paths['img_params_500'])
    scanner = gc.GCScannerOwned(util_paths['SAVANT_json'])
    dataset = gc.GCListModeLUTOwned(
        scanner, dataset_paths['test_osem_siddon_multi_ray'])
    sens_img = gc.GCImageOwned(
        img_params, util_paths['sens_SAVANT_multi_ray_500'])

    _helper._test_reconstruction(
        img_params, scanner, dataset, sens_img,
        out_paths['test_osem_siddon_multi_ray'],
        ref_paths['test_osem_siddon_multi_ray'],
        num_MLEM_iterations=3, num_OSEM_subsets=12, num_rays=num_siddon_rays)



# %% Standalone command line

if __name__ == '__main__':
    print('Run \'pytest test_recon.py\' to launch integration tests')
