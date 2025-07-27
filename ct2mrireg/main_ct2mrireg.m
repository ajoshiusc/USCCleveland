%% Intra-modal rigid registration example
clc; close all; clear;
restoredefaultpath;      % To remove conflicting libraries. May remove this line.

% mandatory inputs
fileDir = '/home/ajoshi/Downloads';
%ct_img = fullfile(fileDir, 'F1998H93_CT.nii'); % CT image
%mr_img = fullfile(fileDir, 'F1998H93_preMRI.bse.nii.gz'); % MR image
out_img = fullfile(fileDir, 'F1998H93_CT_bse_reg.nii'); %Output Image

mr_img = '/home/ajoshi/Downloads/ct2mri_fail_data/MRI.nii.gz';
ct_img ='/home/ajoshi/Downloads/ct2mri_fail_data/CT.nii.gz';
out_img = '/home/ajoshi/Downloads/ct2mri_fail_data/CT2MRI.nii.gz';

%mr_img ='/home/ajoshi/Desktop/ucdata/UG_data/MRI/11_c_Ax_T1_Stealth_Bravo_ALL_8_BRAIN_ADULT_20230114123955_11.nii';;
%ct_img = '/home/ajoshi/Desktop/ucdata/UG_data/CT/3_THIN_BONE_1.2_BRAIN-HELICAL_SCAN_WITH_ANGLED_AXIAL_REFORMATS_20230123135529_3_corr.nii';
%out_img = '/home/ajoshi/Desktop/ucdata/UG_data/MRI/11_c_Ax_T1_Stealth_Bravo_ALL_8_BRAIN_ADULT_20230114123955_11_ct.nii.gz';
% Register CT (moving) to MR (fixed)

a=tic;
ct2mrireg(ct_img, mr_img, out_img);
toc(a)

