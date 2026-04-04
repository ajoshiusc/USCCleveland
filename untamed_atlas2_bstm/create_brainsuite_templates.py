#!/usr/bin/env python3
"""
create_brainsuite_templates.py
------------------------------
Converts the Untamed parcellations into templates compatible with BrainSuite (SVReg).

Outputs:
1. FreeSurfer-style `.annot` files for `fsaverage6` space.
2. CIFTI `.dlabel.nii` files for `fs_LR 32k` space.

Author: auto-generated
"""

import os
import numpy as np
import scipy.io
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings

warnings.filterwarnings("ignore") # suppress pixdim warnings

# ─── Paths ────────────────────────────────────────────────────────────────────

ATLAS_DIR_FS6  = "/home/ajoshi/Downloads/24750117/fsaverage6"
ATLAS_DIR_32K  = "/home/ajoshi/Downloads/24750117/fsLR_32K"

FS6_LABEL_DIR  = "/home/ajoshi/Software/freesurfer_u22/subjects/fsaverage6/label"

LR32K_DIR = (
    "/home/ajoshi/Software/BrainSuite23a/svreg.new/USCBrainMulti/orig_atlas/"
    "Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/"
    "MNINonLinear/fsaverage_LR32k"
)

# Reference HCP dlabel files to clone their geometry
REF_CIFT_L = os.path.join(LR32K_DIR, "Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii")
REF_CIFT_R = os.path.join(LR32K_DIR, "Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii")

OUTPUT_DIR = "brainsuite_templates"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fsaverage6"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "fsLR32k"), exist_ok=True)

FS6_ATLASES = {
    "L26R25": os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L26R25.mat"),
    "L50R50": os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L50R50.mat"),
    "L100R100": os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L100R100.mat"),
    "L180R180": os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L180R180.mat"),
    "L200R200": os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L200R200.mat"),
}

LR32K_ATLASES = {
    "L26R25": os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L26R25.mat"),
    "L50R50": os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L50R50.mat"),
    "L100R100": os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L100R100.mat"),
    "L180R180": os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L180R180.mat"),
    "L200R200": os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L200R200.mat"),
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_label_cmap(n_labels):
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    colors = [base_colors[i % len(base_colors)] for i in range(n_labels + 1)]
    return np.array(colors)

def load_fs6_cortex_mask():
    lh_cortex = np.sort(nib.freesurfer.read_label(os.path.join(FS6_LABEL_DIR, "lh.cortex.label")))
    rh_cortex = np.sort(nib.freesurfer.read_label(os.path.join(FS6_LABEL_DIR, "rh.cortex.label")))
    return lh_cortex, rh_cortex

def load_32k_cortex_mask():
    cifti = nib.load(os.path.join(LR32K_DIR, "Q1-Q6_RelatedParcellation210.corrThickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"))
    lh_idx, rh_idx = None, None
    for bm in cifti.header.get_index_map(1).brain_models:
        if bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            lh_idx = np.array(bm.vertex_indices)
        elif bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            rh_idx = np.array(bm.vertex_indices)
    return lh_idx, rh_idx

def unpack_labels(label_flat, lh_cortex_idx, rh_cortex_idx, n_lh, n_rh):
    n_lh_cortex = len(lh_cortex_idx)
    lh_label_vec = np.zeros(n_lh, dtype=np.int32)
    rh_label_vec = np.zeros(n_rh, dtype=np.int32)
    lh_label_vec[lh_cortex_idx] = label_flat[:n_lh_cortex].astype(np.int32)
    rh_label_vec[rh_cortex_idx] = label_flat[n_lh_cortex:].astype(np.int32)
    return lh_label_vec, rh_label_vec

# ─── Exporters ───────────────────────────────────────────────────────────────

def export_annot(filepath, labels_vec, cmap_colors, hemi="L"):
    n_colors = len(cmap_colors)
    ctab = np.zeros((n_colors, 5), dtype=np.int32)
    names = [b"unknown"]
    
    ctab[0] = [0, 0, 0, 0, 0]
    for lbl in range(1, n_colors):
        r, g, b = (cmap_colors[lbl][:3] * 255).astype(int)
        ctab[lbl] = [r, g, b, 0, lbl]
        names.append(f"Untamed_{hemi}{lbl}".encode("utf-8"))

    nib.freesurfer.write_annot(filepath, labels_vec, ctab, names)

def export_dlabel(filepath, ref_cifti_path, labels_vec, cmap_colors, hemi="L"):
    # Load reference template to get proper geometry headers
    ref_cifti = nib.load(ref_cifti_path)
    
    # Rebuild dictionary with our custom colors
    new_dict = {0: ("???", (0.0, 0.0, 0.0, 0.0))}
    n_colors = len(cmap_colors)
    
    for lbl in range(1, n_colors):
        r, g, b = cmap_colors[lbl][:3]
        new_dict[lbl] = (f"Untamed_{hemi}{lbl}", (float(r), float(g), float(b), 1.0))
        
    old_axis = ref_cifti.header.get_axis(0)
    name = old_axis.name[0]
    
    # Create the new Label mapping axis
    new_label_axis = nib.cifti2.cifti2_axes.LabelAxis([name], [new_dict])
    
    # Build a fresh header using the new map and the old brain geometry map
    new_header = nib.cifti2.Cifti2Header.from_axes((new_label_axis, ref_cifti.header.get_axis(1)))
    
    # Infs_LR32k, the data array in the separate L and R cifti files contains ONLY cortex vertices
    # Our labels_vec actually maps onto all vertices, we must extract the active cortex values back out!
    # Wait: The reference file data shape is (1, N_cortex) 
    # e.g. 29696 for L, 29716 for R.
    n_cortex = ref_cifti.get_fdata().shape[1]
    
    # To re-pack it into CIFTI, we just want the non-zero (or strictly cortex-indexed) labels
    # We can get them by using the raw labels directly, or re-masking them via cortex_mask
    if hemi == "L":
        cortex_mask, _ = load_32k_cortex_mask()
    else:
        _, cortex_mask = load_32k_cortex_mask()

    data_cortex = labels_vec[cortex_mask].astype(np.int32)
    # Ensure they match the dimension of the reference file
    assert len(data_cortex) == n_cortex, "Cortex mask mismatch!"

    new_data = np.zeros((1, n_cortex), dtype=np.int32)
    new_data[0, :] = data_cortex
    
    # Save the custom CIFTI Dense Label
    out_cifti = nib.Cifti2Image(new_data, header=new_header)
    nib.save(out_cifti, filepath)


def main():
    print("Loading fsaverage6 cortex mask")
    lh_idx_fs6, rh_idx_fs6 = load_fs6_cortex_mask()
    n_fs6_lh, n_fs6_rh = 40962, 40962
    
    print("\n── Creating fsaverage6 .annot templates ──")
    for name, mat_path in FS6_ATLASES.items():
        print(f"  {name} …")
        d = scipy.io.loadmat(mat_path)
        labels = d["label"].flatten()
        n_parc = int(labels.max())
        cmap = make_label_cmap(n_parc)
        lh_lbl, rh_lbl = unpack_labels(labels, lh_idx_fs6, rh_idx_fs6, n_fs6_lh, n_fs6_rh)
        
        # Save LH and RH annot
        out_lh = os.path.join(OUTPUT_DIR, "fsaverage6", f"lh.Untamed_{name}.annot")
        out_rh = os.path.join(OUTPUT_DIR, "fsaverage6", f"rh.Untamed_{name}.annot")
        export_annot(out_lh, lh_lbl, cmap, "L")
        export_annot(out_rh, rh_lbl, cmap, "R")
        
    print("Loading fs_LR 32k cortex mask")
    lh_idx_32k, rh_idx_32k = load_32k_cortex_mask()
    n_32k_lh, n_32k_rh = 32492, 32492
    
    print("\n── Creating fs_LR 32k .dlabel.nii templates ──")
    for name, mat_path in LR32K_ATLASES.items():
        print(f"  {name} …")
        d = scipy.io.loadmat(mat_path)
        labels = d["label"].flatten()
        n_parc = int(labels.max())
        cmap = make_label_cmap(n_parc)
        lh_lbl, rh_lbl = unpack_labels(labels, lh_idx_32k, rh_idx_32k, n_32k_lh, n_32k_rh)
        
        out_lh = os.path.join(OUTPUT_DIR, "fsLR32k", f"Untamed_{name}.L.dlabel.nii")
        out_rh = os.path.join(OUTPUT_DIR, "fsLR32k", f"Untamed_{name}.R.dlabel.nii")
        
        export_dlabel(out_lh, REF_CIFT_L, lh_lbl, cmap, "L")
        export_dlabel(out_rh, REF_CIFT_R, rh_lbl, cmap, "R")

    print(f"\nAll BrainSuite templates generated successfully at {OUTPUT_DIR}!")

if __name__ == "__main__":
    main()
