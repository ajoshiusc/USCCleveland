#!/usr/bin/env python3
"""
create_brainstorm_atlases.py
----------------------------
Converts the Untamed parcellations into Brainstorm Scout (.mat) files.

These .mat files can be directly loaded into Brainstorm:
  - Right-click on a cortical surface in Brainstorm.
  - Select 'Scouts > Load' or go to the Scout tab -> Load.

Supports:
  - fsaverage6  (FreeSurfer, ~40962 vertices per hemisphere)
  - fs_LR 32k   (HCP CIFTI space, ~32492 vertices per hemisphere)

Author: auto-generated
"""

import os
import numpy as np
import scipy.io
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ─── Paths ────────────────────────────────────────────────────────────────────

ATLAS_DIR_FS6  = "/home/ajoshi/Downloads/24750117/fsaverage6"
ATLAS_DIR_32K  = "/home/ajoshi/Downloads/24750117/fsLR_32K"

FS6_LABEL_DIR  = "/home/ajoshi/Software/freesurfer_u22/subjects/fsaverage6/label"

LR32K_DIR = (
    "/home/ajoshi/Software/BrainSuite23a/svreg.new/USCBrainMulti/orig_atlas/"
    "Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/"
    "MNINonLinear/fsaverage_LR32k"
)

OUTPUT_DIR = "/home/ajoshi/Projects/TR2preproc/dev/brainstorm_atlases"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FS6_ATLASES = {
    "L26R25"   : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L26R25.mat"),
    "L50R50"   : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L50R50.mat"),
    "L100R100" : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L100R100.mat"),
    "L180R180" : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L180R180.mat"),
    "L200R200" : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L200R200.mat"),
}

LR32K_ATLASES = {
    "L26R25"   : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L26R25.mat"),
    "L50R50"   : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L50R50.mat"),
    "L100R100" : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L100R100.mat"),
    "L180R180" : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L180R180.mat"),
    "L200R200" : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L200R200.mat"),
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_label_cmap(n_labels):
    """Return a categorical colormap array for n_labels parcels."""
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    colors = [base_colors[i % len(base_colors)] for i in range(n_labels + 1)]
    return np.array(colors)

def load_fs6_cortex_mask():
    lh_cortex = np.sort(nib.freesurfer.read_label(os.path.join(FS6_LABEL_DIR, "lh.cortex.label")))
    rh_cortex = np.sort(nib.freesurfer.read_label(os.path.join(FS6_LABEL_DIR, "rh.cortex.label")))
    return lh_cortex, rh_cortex

def load_32k_cortex_mask():
    cifti_path = os.path.join(
        LR32K_DIR,
        "Q1-Q6_RelatedParcellation210.corrThickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"
    )
    cifti = nib.load(cifti_path)
    lh_idx = None
    rh_idx = None
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

# ─── Brainstorm Scout File Generator ─────────────────────────────────────────

def create_brainstorm_atlas_scouts(atlas_name, lh_labels, rh_labels, cmap_colors, out_path, n_lh_vertices):
    """
    Creates a valid Brainstorm scout .mat structure and saves it.
    
    Brainstorm indexing:
    - Vertices are 1-based.
    - Right hemisphere vertices are typically offset by n_lh_vertices in brainstorm 
      for a combined cortex! 
    Wait, brainstorm handles both hemispheres together as a single surface (e.g., tess_cortex_pial_low.mat)
    where vertices [0 ... n_lh-1] are LH and [n_lh ... n_lh+n_rh-1] are RH.
    So right hemisphere indices in the Scout must be offset by `n_lh_vertices`.
    """
    
    # Brainstorm expects a single struct containing "Name" and "Scouts"
    # Numpy constructs this via structured arrays. 
    scout_dtype = [
        ('Vertices', 'O'),
        ('Seed', 'O'),
        ('Color', 'O'),
        ('Label', 'O'),
        ('Function', 'O'),
        ('Region', 'O'),
        ('Handles', 'O')
    ]
    
    # Extract unique labels > 0
    unique_lh = np.unique(lh_labels[lh_labels > 0])
    unique_rh = np.unique(rh_labels[rh_labels > 0])
    num_scouts = len(unique_lh) + len(unique_rh)
    
    scouts_array = np.zeros((num_scouts,), dtype=scout_dtype)
    
    scout_idx = 0
    
    # Process Left Hemisphere
    for lbl in unique_lh:
        # Match vertices, Brainstorm uses 1-based indexing
        verts = np.where(lh_labels == lbl)[0] + 1
        
        scouts_array[scout_idx]['Vertices'] = verts.astype(float)
        scouts_array[scout_idx]['Seed'] = float(verts[0])
        scouts_array[scout_idx]['Color'] = cmap_colors[lbl].astype(float)[:3]
        scouts_array[scout_idx]['Label'] = f"Untamed_L{lbl}"
        scouts_array[scout_idx]['Function'] = "Mean"
        scouts_array[scout_idx]['Region'] = "LU"  # Left unknown lobe
        scouts_array[scout_idx]['Handles'] = np.array([], dtype=float)
        
        scout_idx += 1
        
    # Process Right Hemisphere
    for lbl in unique_rh:
        # Match vertices, 1-based, and offset by number of LH vertices
        verts = np.where(rh_labels == lbl)[0] + 1 + n_lh_vertices
        
        scouts_array[scout_idx]['Vertices'] = verts.astype(float)
        scouts_array[scout_idx]['Seed'] = float(verts[0])
        scouts_array[scout_idx]['Color'] = cmap_colors[lbl].astype(float)[:3]
        scouts_array[scout_idx]['Label'] = f"Untamed_R{lbl}"
        scouts_array[scout_idx]['Function'] = "Mean"
        scouts_array[scout_idx]['Region'] = "RU"  # Right unknown lobe
        scouts_array[scout_idx]['Handles'] = np.array([], dtype=float)
        
        scout_idx += 1

    # Create root atlas struct
    atlas_dtype = [('Name', 'O'), ('Scouts', 'O')]
    atlas_array = np.zeros((1,), dtype=atlas_dtype)
    atlas_array[0]['Name'] = f"Untamed Atlas {atlas_name}"
    atlas_array[0]['Scouts'] = scouts_array
    
    # Save to standard scout format
    scipy.io.savemat(out_path, {'Atlas': atlas_array}, do_compression=True)
    print(f"  Saved Brainstorm Atlas -> {out_path}")


def main():
    print("Loading fsaverage6 cortex mask")
    lh_idx_fs6, rh_idx_fs6 = load_fs6_cortex_mask()

    print("Loading fs_LR 32k cortex mask")
    lh_idx_32k, rh_idx_32k = load_32k_cortex_mask()

    n_fs6_lh = 40962
    n_fs6_rh = 40962
    
    n_32k_lh = 32492
    n_32k_rh = 32492

    print("\n── Processing fsaverage6 atlases ──")
    for name, mat_path in FS6_ATLASES.items():
        print(f"  {name} …")
        d = scipy.io.loadmat(mat_path)
        labels = d["label"].flatten()
        n_parc = int(labels.max())
        cmap = make_label_cmap(n_parc)
        
        lh_lbl, rh_lbl = unpack_labels(labels, lh_idx_fs6, rh_idx_fs6, n_fs6_lh, n_fs6_rh)
        
        out_path = os.path.join(OUTPUT_DIR, f"scout_Untamed_{name}_fsaverage6.mat")
        create_brainstorm_atlas_scouts(name, lh_lbl, rh_lbl, cmap, out_path, n_fs6_lh)

    print("\n── Processing fs_LR 32k atlases ──")
    for name, mat_path in LR32K_ATLASES.items():
        print(f"  {name} …")
        d = scipy.io.loadmat(mat_path)
        labels = d["label"].flatten()
        n_parc = int(labels.max())
        cmap = make_label_cmap(n_parc)
        
        lh_lbl, rh_lbl = unpack_labels(labels, lh_idx_32k, rh_idx_32k, n_32k_lh, n_32k_rh)
        
        out_path = os.path.join(OUTPUT_DIR, f"scout_Untamed_{name}_fsLR32k.mat")
        create_brainstorm_atlas_scouts(name, lh_lbl, rh_lbl, cmap, out_path, n_32k_lh)

    print("\nAll Brainstorm scout files generated successfully!")

if __name__ == "__main__":
    main()
