#!/usr/bin/env python3
"""
create_brainstorm_default_anatomy.py
------------------------------------
Creates fully standalone "Default Anatomy" Brainstorm directories for fsaverage6 
and fsLR32k spaces containing the Untamed parcellations natively embedded as Scout atlases.

These directories contain:
1. `brainstormsubject.mat` - defines the anatomy protocol.
2. `tess_cortex.mat` - contains the joined 3D cortical surfaces (Vertices, Faces) 
   and all the Untamed Atlas structs natively embedded into the surface.

Author: auto-generated
"""

import os
import numpy as np
import scipy.io
import nibabel as nib
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────

ATLAS_DIR_FS6  = "/home/ajoshi/Downloads/24750117/fsaverage6"
ATLAS_DIR_32K  = "/home/ajoshi/Downloads/24750117/fsLR_32K"
FS6_BASE_DIR   = "/home/ajoshi/Software/freesurfer_u22/subjects/fsaverage6"

LR32K_DIR = (
    "/home/ajoshi/Software/BrainSuite23a/svreg.new/USCBrainMulti/orig_atlas/"
    "Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/"
    "MNINonLinear/fsaverage_LR32k"
)

OUT_DIR = "/home/ajoshi/Projects/TR2preproc/dev/brainstorm_default_anatomies"
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def make_label_cmap(n_labels):
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    return np.array([base_colors[i % len(base_colors)] for i in range(n_labels + 1)])

def load_fs6_surfaces():
    lh_c, lh_f = nib.freesurfer.read_geometry(os.path.join(FS6_BASE_DIR, "surf", "lh.inflated"))
    rh_c, rh_f = nib.freesurfer.read_geometry(os.path.join(FS6_BASE_DIR, "surf", "rh.inflated"))
    lh_idx = np.sort(nib.freesurfer.read_label(os.path.join(FS6_BASE_DIR, "label", "lh.cortex.label")))
    rh_idx = np.sort(nib.freesurfer.read_label(os.path.join(FS6_BASE_DIR, "label", "rh.cortex.label")))
    return lh_c, lh_f, rh_c, rh_f, lh_idx, rh_idx

def load_32k_surfaces():
    lh_gii = nib.load(os.path.join(LR32K_DIR, "Q1-Q6_RelatedParcellation210.L.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii"))
    rh_gii = nib.load(os.path.join(LR32K_DIR, "Q1-Q6_RelatedParcellation210.R.inflated_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.surf.gii"))
    lh_c, lh_f = lh_gii.darrays[0].data, lh_gii.darrays[1].data
    rh_c, rh_f = rh_gii.darrays[0].data, rh_gii.darrays[1].data
    
    cifti = nib.load(os.path.join(LR32K_DIR, "Q1-Q6_RelatedParcellation210.corrThickness_MSMAll_2_d41_WRN_DeDrift.32k_fs_LR.dscalar.nii"))
    lh_idx, rh_idx = None, None
    for bm in cifti.header.get_index_map(1).brain_models:
        if bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_LEFT":
            lh_idx = np.array(bm.vertex_indices)
        elif bm.brain_structure == "CIFTI_STRUCTURE_CORTEX_RIGHT":
            rh_idx = np.array(bm.vertex_indices)
    return lh_c, lh_f, rh_c, rh_f, lh_idx, rh_idx

def unpack_labels(label_flat, lh_cortex_idx, rh_cortex_idx, n_lh, n_rh):
    n_lh_cortex = len(lh_cortex_idx)
    lh_lbl = np.zeros(n_lh, dtype=np.int32)
    rh_lbl = np.zeros(n_rh, dtype=np.int32)
    lh_lbl[lh_cortex_idx] = label_flat[:n_lh_cortex].astype(np.int32)
    rh_lbl[rh_cortex_idx] = label_flat[n_lh_cortex:].astype(np.int32)
    return lh_lbl, rh_lbl

# ─── Builders ────────────────────────────────────────────────────────────────

def build_anatomy(folder_name, lh_c, lh_f, rh_c, rh_f, lh_idx, rh_idx, atlases_dict):
    anatomy_dir = os.path.join(OUT_DIR, folder_name)
    os.makedirs(anatomy_dir, exist_ok=True)
    
    n_lh = lh_c.shape[0]
    n_rh = rh_c.shape[0]
    
    # Scale surfaces to meters (0.001) for Brainstorm
    # fsaverage6 inflated hemispheres natively overlap completely on the X axis [-41, 41].
    # fsLR32k natives correctly span the full 130mm range.
    # We explicitly shift fsaverage6 apart so it behaves geometrically like 32k.
    if "fsaverage6" in folder_name:
        lh_c = lh_c.copy()
        lh_c[:, 0] -= 45.0
        rh_c = rh_c.copy()
        rh_c[:, 0] += 45.0
        
    vertices = np.vstack([lh_c, rh_c]) * 0.001
    
    # Brainstorm / MATLAB rendering uses the opposite face winding order 
    # to FreeSurfer and GIFTI standard. We reverse the order [0, 2, 1]
    # to flip the normals outward, fixing the "edge only" coloring bug.
    
    # LH Faces: 1-indexed for matlab, reversed winding
    faces_lh = lh_f[:, [0, 2, 1]] + 1
    # RH Faces: 1-indexed AND shifted by n_lh, reversed winding
    faces_rh = rh_f[:, [0, 2, 1]] + 1 + n_lh
    faces = np.vstack([faces_lh, faces_rh])
    
    # 2. Extract atlas scouts
    atlas_dtype = [('Name', 'O'), ('Scouts', 'O')]
    scout_dtype = [('Vertices', 'O'), ('Seed', 'O'), ('Color', 'O'), ('Label', 'O'), ('Function', 'O'), ('Region', 'O'), ('Handles', 'O')]
    
    all_atlases = np.zeros((len(atlases_dict),), dtype=atlas_dtype)
    
    print(f"Creating {folder_name} ...")
    
    for a_idx, (name, mat_path) in enumerate(atlases_dict.items()):
        d = scipy.io.loadmat(mat_path)
        labels = d["label"].flatten()
        n_parc = int(labels.max())
        cmap = make_label_cmap(n_parc)
        
        lh_lbl, rh_lbl = unpack_labels(labels, lh_idx, rh_idx, n_lh, n_rh)
        
        unique_lh = np.unique(lh_lbl[lh_lbl > 0])
        unique_rh = np.unique(rh_lbl[rh_lbl > 0])
        num_scouts = len(unique_lh) + len(unique_rh)
        
        scouts_array = np.zeros((num_scouts,), dtype=scout_dtype)
        s_idx = 0
        
        for lbl in unique_lh:
            verts = np.where(lh_lbl == lbl)[0] + 1
            scouts_array[s_idx]['Vertices'] = verts.astype(float)
            scouts_array[s_idx]['Seed'] = float(verts[0])
            scouts_array[s_idx]['Color'] = cmap[lbl].astype(float)[:3]
            scouts_array[s_idx]['Label'] = f"Untamed_L{lbl}"
            scouts_array[s_idx]['Function'] = "Mean"
            scouts_array[s_idx]['Region'] = "LU"
            scouts_array[s_idx]['Handles'] = np.array([], dtype=float)
            s_idx += 1
            
        for lbl in unique_rh:
            verts = np.where(rh_lbl == lbl)[0] + 1 + n_lh
            scouts_array[s_idx]['Vertices'] = verts.astype(float)
            scouts_array[s_idx]['Seed'] = float(verts[0])
            scouts_array[s_idx]['Color'] = cmap[lbl].astype(float)[:3]
            scouts_array[s_idx]['Label'] = f"Untamed_R{lbl}"
            scouts_array[s_idx]['Function'] = "Mean"
            scouts_array[s_idx]['Region'] = "RU"
            scouts_array[s_idx]['Handles'] = np.array([], dtype=float)
            s_idx += 1
            
        all_atlases[a_idx]['Name'] = f"Untamed Atlas {name}"
        all_atlases[a_idx]['Scouts'] = scouts_array
        
    # Write tess_cortex.mat
    tess_path = os.path.join(anatomy_dir, "tess_cortex.mat")
    tess_struct = {
        'Comment': 'Cortex_Untamed',
        'Vertices': vertices,
        'Faces': faces,
        'Atlas': all_atlases
    }
    scipy.io.savemat(tess_path, tess_struct, do_compression=True)
    
    # Write brainstormsubject.mat
    bs_path = os.path.join(anatomy_dir, "brainstormsubject.mat")
    bs_struct = {
        'Comments': [],
        'Anatomy': '',
        'UseDefaultAnat': 0,
        'UseDefaultChannel': 0,
        'Scalp': '',
        'InnerSkull': '',
        'OuterSkull': '',
        'Fibers': [],
        'FEM': [],
        'Cortex': f"{folder_name}/tess_cortex.mat"
    }
    scipy.io.savemat(bs_path, bs_struct)
    
    print(f"  Saved Anatomy Template to {anatomy_dir}")


def main():
    print("Loading fsaverage6 surfaces...")
    lh_c, lh_f, rh_c, rh_f, lh_i, rh_i = load_fs6_surfaces()
    FS6 = {
        "L26R25"   : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L26R25.mat"),
        "L50R50"   : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L50R50.mat"),
        "L100R100" : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L100R100.mat"),
        "L180R180" : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L180R180.mat"),
        "L200R200" : os.path.join(ATLAS_DIR_FS6, "Untamed_fs6_L200R200.mat"),
    }
    build_anatomy("Untamed_fsaverage6", lh_c, lh_f, rh_c, rh_f, lh_i, rh_i, FS6)

    print("Loading fsLR32k surfaces...")
    lh_c, lh_f, rh_c, rh_f, lh_i, rh_i = load_32k_surfaces()
    LR32K = {
        "L26R25"   : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L26R25.mat"),
        "L50R50"   : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L50R50.mat"),
        "L100R100" : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L100R100.mat"),
        "L180R180" : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L180R180.mat"),
        "L200R200" : os.path.join(ATLAS_DIR_32K, "Untamed_fsLR32K_L200R200.mat"),
    }
    build_anatomy("Untamed_fsLR32k", lh_c, lh_f, rh_c, rh_f, lh_i, rh_i, LR32K)

if __name__ == "__main__":
    main()
