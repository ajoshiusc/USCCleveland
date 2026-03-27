#!/usr/bin/env python3
"""
plot_untamed_atlas_pyvista.py
----------------------------
Plot the Untamed atlas parcellations using PyVista (interactive).

Supports two surface spaces:
  - fsaverage6  (FreeSurfer, ~40962 vertices per hemisphere)
  - fs_LR 32k   (HCP CIFTI space, ~32492 vertices per hemisphere)

Usage:
    source .venv/bin/activate
    python plot_untamed_atlas_pyvista.py

Author: auto-generated
"""

import os
import numpy as np
import scipy.io
import nibabel as nib
import pyvista as pv
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

# ─── Paths ────────────────────────────────────────────────────────────────────

ATLAS_DIR_FS6  = "/home/ajoshi/Downloads/24750117/fsaverage6"
ATLAS_DIR_32K  = "/home/ajoshi/Downloads/24750117/fsLR_32K"

FS6_SURF_DIR   = "/home/ajoshi/Software/freesurfer_u22/subjects/fsaverage6/surf"
FS6_LABEL_DIR  = "/home/ajoshi/Software/freesurfer_u22/subjects/fsaverage6/label"

LR32K_DIR = (
    "/home/ajoshi/Software/BrainSuite23a/svreg.new/USCBrainMulti/orig_atlas/"
    "Glasser_et_al_2016_HCP_MMP1.0_RVVG/HCP_PhaseTwo/Q1-Q6_RelatedParcellation210/"
    "MNINonLinear/fsaverage_LR32k"
)

# ─── Atlas files to visualise ──────────────────────────

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

# ─── Colourmap ─────────────────────────────────────────

def make_label_cmap(n_labels):
    """Return a categorical colormap for n_labels parcels."""
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    colors = [base_colors[i % len(base_colors)] for i in range(n_labels + 1)]
    colors[0] = (0.15, 0.15, 0.15)  # label 0 → dark background
    return ListedColormap(colors)

# ─── Helpers: load surfaces ────────────────────────────────────────

def load_fs6_surfaces(surf_type="inflated"):
    """Returns (lh_coords, lh_faces, rh_coords, rh_faces) for fsaverage6."""
    lh_coords, lh_faces = nib.freesurfer.read_geometry(
        os.path.join(FS6_SURF_DIR, f"lh.{surf_type}")
    )
    rh_coords, rh_faces = nib.freesurfer.read_geometry(
        os.path.join(FS6_SURF_DIR, f"rh.{surf_type}")
    )
    return lh_coords, lh_faces, rh_coords, rh_faces

def load_fs6_cortex_mask():
    lh_cortex = np.sort(nib.freesurfer.read_label(
        os.path.join(FS6_LABEL_DIR, "lh.cortex.label")
    ))
    rh_cortex = np.sort(nib.freesurfer.read_label(
        os.path.join(FS6_LABEL_DIR, "rh.cortex.label")
    ))
    return lh_cortex, rh_cortex

def load_32k_surfaces(surf_type="inflated"):
    type_map = {
        "inflated"   : "inflated_MSMAll_2_d41_WRN_DeDrift",
        "pial"       : "pial_MSMAll_2_d41_WRN_DeDrift",
        "midthick"   : "midthickness_MSMAll_2_d41_WRN_DeDrift",
        "veryinflated": "very_inflated_MSMAll_2_d41_WRN_DeDrift",
    }
    tag = type_map.get(surf_type, surf_type)
    lh_gii = nib.load(os.path.join(
        LR32K_DIR,
        f"Q1-Q6_RelatedParcellation210.L.{tag}.32k_fs_LR.surf.gii"
    ))
    rh_gii = nib.load(os.path.join(
        LR32K_DIR,
        f"Q1-Q6_RelatedParcellation210.R.{tag}.32k_fs_LR.surf.gii"
    ))
    lh_coords = lh_gii.darrays[0].data
    lh_faces  = lh_gii.darrays[1].data
    rh_coords = rh_gii.darrays[0].data
    rh_faces  = rh_gii.darrays[1].data
    return lh_coords, lh_faces, rh_coords, rh_faces

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

# ─── Map atlas labels onto full surface vertex array ─────────────────────────

def unpack_fs6_labels(label_flat, lh_cortex_idx, rh_cortex_idx, n_lh=40962, n_rh=40962):
    n_lh_cortex = len(lh_cortex_idx)
    lh_label_vec = np.zeros(n_lh, dtype=np.int32)
    rh_label_vec = np.zeros(n_rh, dtype=np.int32)
    lh_label_vec[lh_cortex_idx] = label_flat[:n_lh_cortex].astype(np.int32)
    rh_label_vec[rh_cortex_idx] = label_flat[n_lh_cortex:].astype(np.int32)
    return lh_label_vec, rh_label_vec

def unpack_32k_labels(label_flat, lh_cortex_idx, rh_cortex_idx, n_lh=32492, n_rh=32492):
    n_lh_cortex = len(lh_cortex_idx)
    lh_label_vec = np.zeros(n_lh, dtype=np.int32)
    rh_label_vec = np.zeros(n_rh, dtype=np.int32)
    lh_label_vec[lh_cortex_idx] = label_flat[:n_lh_cortex].astype(np.int32)
    rh_label_vec[rh_cortex_idx] = label_flat[n_lh_cortex:].astype(np.int32)
    return lh_label_vec, rh_label_vec

# ─── PyVista Helpers ─────────────────────────────────────────────────────────

def nib_to_pyvista(coords, faces):
    """Convert nibabel geometry to pyvista PolyData."""
    # Faces in nibabel are (N, 3), in pv they need [3, v1, v2, v3, 3, v4, v5, v6, ...]
    pv_faces = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return pv.PolyData(coords, pv_faces)

# ─── High-level plot function ─────────────────────────────────────────────────

def plot_atlas_pyvista(atlas_name, space,
                       lh_coords, lh_faces, rh_coords, rh_faces,
                       lh_labels, rh_labels, n_parcels):
    """
    Create an interactive 4-view plot with PyVista.
    """
    cmap = make_label_cmap(n_parcels)
    
    # Pre-compute cell labels for solid coloring (prevents boundary interpolation artifacts)
    def compute_cell_labels(faces, point_labels):
        v0 = point_labels[faces[:, 0]]
        v1 = point_labels[faces[:, 1]]
        v2 = point_labels[faces[:, 2]]
        # Use majority vote across face vertices
        return np.where(v0 == v1, v0, np.where(v0 == v2, v0, v1))

    lh_cell_labels = compute_cell_labels(lh_faces, lh_labels)
    rh_cell_labels = compute_cell_labels(rh_faces, rh_labels)
    
    lh_mesh = nib_to_pyvista(lh_coords, lh_faces)
    lh_mesh.cell_data["labels"] = lh_cell_labels
    
    rh_mesh = nib_to_pyvista(rh_coords, rh_faces)
    rh_mesh.cell_data["labels"] = rh_cell_labels
    
    plotter = pv.Plotter(shape=(1, 4), title=f"Untamed Atlas - {atlas_name} ({space})")
    plotter.set_background("black")
    
    # Common plotting arguments
    # Note: Using scalars="labels" with cell data ensures each face has a solid color.
    mesh_args = dict(scalars="labels", cmap=cmap, show_scalar_bar=False, smooth_shading=True, clim=[0, n_parcels], preference="cell")
    
    # 1. LH Lateral
    plotter.subplot(0, 0)
    plotter.add_text("LH Lateral", position="upper_left", font_size=10, color="white")
    plotter.add_mesh(lh_mesh, **mesh_args)
    plotter.view_yz(negative=True) # View from Left
    
    # 2. LH Medial
    plotter.subplot(0, 1)
    plotter.add_text("LH Medial", position="upper_left", font_size=10, color="white")
    plotter.add_mesh(lh_mesh, **mesh_args)
    plotter.view_yz(negative=False) # View from Right (shows medial for LH)
    
    # 3. RH Medial
    plotter.subplot(0, 2)
    plotter.add_text("RH Medial", position="upper_left", font_size=10, color="white")
    plotter.add_mesh(rh_mesh, **mesh_args)
    plotter.view_yz(negative=True) # View from Left (shows medial for RH)
    
    # 4. RH Lateral
    plotter.subplot(0, 3)
    plotter.add_text("RH Lateral", position="upper_left", font_size=10, color="white")
    plotter.add_mesh(rh_mesh, **mesh_args)
    plotter.view_yz(negative=False) # View from Right
    
    # Link all cameras
    plotter.link_views()
    
    print(f"Showing interactive plot for {atlas_name} ({space}) ...")
    plotter.show()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Allow user to pick which space to plot
    print("Which space would you like to plot?")
    print("  1. fsaverage6")
    print("  2. fs_LR 32k")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        print("Loading fsaverage6 surfaces …")
        lh_coords, lh_faces, rh_coords, rh_faces = load_fs6_surfaces("inflated")
        lh_cortex_idx, rh_cortex_idx = load_fs6_cortex_mask()
        atlases = FS6_ATLASES
        space = "fsaverage6"
        unpack_fn = unpack_fs6_labels
    elif choice == "2":
        print("Loading fs_LR 32k surfaces …")
        lh_coords, lh_faces, rh_coords, rh_faces = load_32k_surfaces("inflated")
        lh_idx, rh_idx = load_32k_cortex_mask()
        atlases = LR32K_ATLASES
        space = "fsLR32k"
        unpack_fn = unpack_32k_labels
        lh_cortex_idx, rh_cortex_idx = lh_idx, rh_idx
    else:
        print("Invalid choice. Exiting.")
        return

    print(f"\nAvailable {space} atlases:")
    for i, name in enumerate(atlases.keys()):
        print(f"  {i+1}. {name}")
    
    atlas_choice = int(input(f"Pick an atlas (1-{len(atlases)}): ")) - 1
    atlas_name = list(atlases.keys())[atlas_choice]
    mat_path = atlases[atlas_name]
    
    print(f"Loading {atlas_name} …")
    d       = scipy.io.loadmat(mat_path)
    labels  = d["label"].flatten()
    n_parc  = int(labels.max())
    
    lh_lbl, rh_lbl = unpack_fn(
        labels, lh_cortex_idx, rh_cortex_idx,
        n_lh=lh_coords.shape[0], n_rh=rh_coords.shape[0]
    )
    
    plot_atlas_pyvista(
        atlas_name, space,
        lh_coords, lh_faces, rh_coords, rh_faces,
        lh_lbl, rh_lbl, n_parc,
    )

if __name__ == "__main__":
    main()
