import nilearn.image as ni
from warp_utils import apply_warp
from aligner import Aligner
from monai.transforms import (
    LoadImage,
    Resize,
    EnsureChannelFirst,
    ScaleIntensityRangePercentiles,
)
import nibabel as nib
import os
import os.path
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.morphology import remove_small_objects, opening


BrainSuitePATH = "/home/ajoshi/BrainSuite23a"
ERR_THR = 80

rigid_reg = Aligner()


mov_img_orig = "/deneb_disk/auto_resection/Ken_Post-op_MRI/sub-SUB102/sMRI/sub-SUB102-102_MRI.nii"
mov_img = "/deneb_disk/auto_resection/Ken_Post-op_MRI/sub-SUB102/sMRI/sub-SUB102-102_MRI.bse.nii.gz"
ref_img = "/home/ajoshi/BrainSuite21a/svreg/BrainSuiteAtlas1/mri.pvc.frac.nii.gz"
ref_img_mask = "/home/ajoshi/BrainSuite21a/svreg/BrainSuiteAtlas1/mri.mask.nii.gz"
ref_img_pvc_frac = "/home/ajoshi/BrainSuite21a/svreg/BrainSuiteAtlas1/mri.pvc.frac.nii.gz"
error_img = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/error_pre_post.nii.gz"
error_mask_img = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/error_pre_post.mask.nii.gz"
error_init_mask_img = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/error_pre_post.init.mask.nii.gz"


# rigidly warped image
rigid_reg_img = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/post2pre.nii.gz"
rigid_reg_img_bse = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/post2pre.bse.nii.gz"
rigid_reg_img_mask = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/post2pre.mask.nii.gz"
rigid_reg_img_bfc = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/post2pre.bfc.nii.gz"
rigid_reg_img_pvc_label = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/post2pre.pvc.label.nii.gz"
rigid_reg_img_pvc_frac = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/post2pre.pvc.frac.nii.gz"

ddf = "/deneb_disk/auto_resection/Andrew_Pre-op_MRI_and_EZ_Map/Subject102/ddf.nii.gz"


rigid_reg.affine_reg(
    fixed_file=ref_img,
    moving_file=mov_img,
    output_file=rigid_reg_img_bse,
    ddf_file=ddf,
    loss="mse",
    device="cuda",
)


moving, moving_meta = LoadImage()(mov_img_orig)
moving = EnsureChannelFirst()(moving)

target, _ = LoadImage()(ref_img)
target = EnsureChannelFirst()(target)


image_movedo = apply_warp(rigid_reg.ddf[None,], moving[None,], target[None,])

nib.save(
    nib.Nifti1Image(image_movedo[0, 0].detach().cpu().numpy(), rigid_reg.target.affine),
    rigid_reg_img,
)


cmd = (
    os.path.join(BrainSuitePATH, "bin", "bse")
    + " -i "
    + rigid_reg_img
    + " -o "
    + rigid_reg_img_bse
    + " --auto --trim --mask "
    + rigid_reg_img_mask
)
os.system(cmd)

cmd = (
    os.path.join(BrainSuitePATH, "bin", "bfc")
    + " -i "
    + rigid_reg_img
    + " -o "
    + rigid_reg_img_bfc
    + " --iterate -m "
    + rigid_reg_img_mask
)
os.system(cmd)


cmd = (
    os.path.join(BrainSuitePATH, "bin", "pvc")
    + " -i "
    + rigid_reg_img_bfc
    + " -o "
    + rigid_reg_img_pvc_label
    + " -f "
    + rigid_reg_img_pvc_frac
)
os.system(cmd)


# Load the images and normalize their intensities
vref, _ = LoadImage()(ref_img_pvc_frac)
vwrp, _ = LoadImage()(rigid_reg_img_pvc_frac)
msk, _ = LoadImage()(ref_img_mask)

vwrp = (255.0 / np.max(vwrp[msk > 0])) * vwrp
vref = (255.0 / np.max(vref[msk > 0])) * vref


# compute the error and smooth the error
vwrp = np.sqrt((vref - vwrp) ** 2)
vwrp = vwrp * (msk > 0)
vwrp = gaussian_filter(vwrp, sigma=1)

nib.save(nib.Nifti1Image(vwrp, rigid_reg.target.affine), error_img)

error_mask = opening(vwrp > ERR_THR)
nib.save(nib.Nifti1Image(255*np.uint8(error_mask), rigid_reg.target.affine), error_init_mask_img)



resection_mask = remove_small_objects(error_mask)
nib.save(
    nib.Nifti1Image(255*np.uint8(resection_mask), rigid_reg.target.affine), error_mask_img
)
