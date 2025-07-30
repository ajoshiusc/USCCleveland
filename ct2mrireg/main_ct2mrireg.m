%% Intra-modal rigid registration example
clc; close all; clear;
restoredefaultpath;      % To remove conflicting libraries. May remove this line.

% mandatory inputs

mr_img = '/home/ajoshi/Downloads/ct2mri_fail_data/MRI.nii.gz';
%'/home/ajoshi/Downloads/ct2mri_fail_data/data2/sub-b086e6e60415_ses-mri3T_acq-MPRAGE_run-01_T1w_1mm.nii.gz';
ct_img ='/home/ajoshi/Downloads/ct2mri_fail_data/CT.nii.gz';
%'/home/ajoshi/Downloads/ct2mri_fail_data/data2/resliced_CT.nii.gz';%sub-b086e6e60415_ses-seeg_run-01_CT_bst.nii.gz';
out_img = '/home/ajoshi/Downloads/ct2mri_fail_data/CT2MR_mi.nii.gz';

%mr_img ='/home/ajoshi/Desktop/ucdata/UG_data/MRI/11_c_Ax_T1_Stealth_Bravo_ALL_8_BRAIN_ADULT_20230114123955_11.nii';;
%ct_img = '/home/ajoshi/Desktop/ucdata/UG_data/CT/3_THIN_BONE_1.2_BRAIN-HELICAL_SCAN_WITH_ANGLED_AXIAL_REFORMATS_20230123135529_3_corr.nii';
%out_img = '/home/ajoshi/Desktop/ucdata/UG_data/MRI/11_c_Ax_T1_Stealth_Bravo_ALL_8_BRAIN_ADULT_20230114123955_11_ct.nii.gz';
% Register CT (moving) to MR (fixed)

a=tic;
ct2mrireg(ct_img, mr_img, out_img);
toc(a)

