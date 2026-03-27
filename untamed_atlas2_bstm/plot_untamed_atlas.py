#!/usr/bin/env python3
"""
plot_untamed_atlas.py
---------------------
Plot the Untamed atlas parcellations as surface renderings.

Supports two surface spaces:
  - fsaverage6  (FreeSurfer, ~40962 vertices per hemisphere)
  - fs_LR 32k   (HCP CIFTI space, ~32492 vertices per hemisphere)

Usage:
    python plot_untamed_atlas.py

Author: auto-generated
"""

import os
import numpy as np
import scipy.io
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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

OUTPUT_DIR = "/home/ajoshi/Projects/TR2preproc/dev/atlas_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Atlas files to visualise (one per subplot row) ──────────────────────────

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

# ─── Colourmap (categorical, cyclic) ─────────────────────────────────────────

def make_label_cmap(n_labels):
    """Return a categorical colormap for n_labels parcels."""
    base_colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors
    colors = [base_colors[i % len(base_colors)] for i in range(n_labels + 1)]
    colors[0] = (0.15, 0.15, 0.15)  # label 0 → dark background
    return mcolors.ListedColormap(colors)


# ─── Helpers: load FreeSurfer surface ────────────────────────────────────────

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
    """
    Return cortex vertex indices for lh and rh in fsaverage6, SORTED.
    The atlas label array has 74947 entries = cortex vertices only,
    stored in ascending vertex-index order.  nibabel.read_label returns
    indices in file order (not necessarily sorted), so we sort them.
    """
    lh_cortex = np.sort(nib.freesurfer.read_label(
        os.path.join(FS6_LABEL_DIR, "lh.cortex.label")
    ))
    rh_cortex = np.sort(nib.freesurfer.read_label(
        os.path.join(FS6_LABEL_DIR, "rh.cortex.label")
    ))
    return lh_cortex, rh_cortex


def load_32k_surfaces(surf_type="inflated"):
    """Returns (lh_coords, lh_faces, rh_coords, rh_faces) for fs_LR 32k."""
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
    """
    Reads the CIFTI header from a 32k dscalar file to get the exact 
    vertex indices that correspond to the left and right cortex.
    """
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
    """
    label_flat : (74947,) array of parcel indices (cortex vertices only)
    Returns lh_labels (n_lh,) and rh_labels (n_rh,) with 0 = medial wall.
    """
    n_lh_cortex = len(lh_cortex_idx)
    lh_label_vec = np.zeros(n_lh, dtype=np.int32)
    rh_label_vec = np.zeros(n_rh, dtype=np.int32)
    lh_label_vec[lh_cortex_idx] = label_flat[:n_lh_cortex].astype(np.int32)
    rh_label_vec[rh_cortex_idx] = label_flat[n_lh_cortex:].astype(np.int32)
    return lh_label_vec, rh_label_vec


def unpack_32k_labels(label_flat, lh_cortex_idx, rh_cortex_idx, n_lh=32492, n_rh=32492):
    """
    For fs_LR 32k the atlas array is typically 59412 = 29696 + 29716 (standard HCP cortex mask).
    Maps the labels to the specific 32k surface vertices using the loaded indices.
    """
    n_lh_cortex = len(lh_cortex_idx)
    lh_label_vec = np.zeros(n_lh, dtype=np.int32)
    rh_label_vec = np.zeros(n_rh, dtype=np.int32)

    # Use the specific vertex_indices from the CIFTI brain models
    lh_label_vec[lh_cortex_idx] = label_flat[:n_lh_cortex].astype(np.int32)
    rh_label_vec[rh_cortex_idx] = label_flat[n_lh_cortex:].astype(np.int32)
    return lh_label_vec, rh_label_vec


# ─── Core rendering (matplotlib trisurf with face-level colour) ──────────────

def compute_face_colors(coords, faces, vertex_labels, cmap_colors):
    """
    Assign each face the colour of the majority label among its 3 vertices.
    Apply diffuse shading from a fixed light direction.
    Returns an (n_faces, 4) RGBA array.
    """
    v0 = vertex_labels[faces[:, 0]]
    v1 = vertex_labels[faces[:, 1]]
    v2 = vertex_labels[faces[:, 2]]
    # Majority vote: prefer the label shared by at least 2 vertices
    face_labels = np.where(v0 == v1, v0, np.where(v0 == v2, v0, v1))
    face_colors = np.array([cmap_colors[lbl] for lbl in face_labels], dtype=float)
    # Diffuse shading: dot product of face normal with light direction
    v0c = coords[faces[:, 0]].astype(float)
    v1c = coords[faces[:, 1]].astype(float)
    v2c = coords[faces[:, 2]].astype(float)
    normals = np.cross(v1c - v0c, v2c - v0c)
    norms   = np.linalg.norm(normals, axis=1, keepdims=True)
    norms   = np.where(norms == 0, 1, norms)
    normals /= norms
    light = np.array([0.0, 0.3, 1.0], dtype=float)
    light /= np.linalg.norm(light)
    shade = np.clip(np.abs(normals @ light), 0, 1) * 0.55 + 0.45
    face_colors[:, :3] *= shade[:, np.newaxis]
    face_colors = np.clip(face_colors, 0, 1)
    return face_colors


def render_hemisphere(ax, coords, faces, vertex_labels, cmap_colors,
                      elev=20, azim=160):
    """
    Render one hemisphere onto a 3D axes using Poly3DCollection.
    This guarantees that each face receives its correct label colour
    (plot_trisurf internally reorders faces for z-sorting, breaking
    per-face colour alignment).
    """
    face_cols = compute_face_colors(coords, faces, vertex_labels, cmap_colors)

    # Build (n_faces, 3, 3) array of triangle vertex coords
    triangles = coords[faces]  # shape: (n_faces, 3, 3)

    poly = Poly3DCollection(
        triangles,
        facecolors=face_cols,
        edgecolors="none",
        zsort="average",
    )
    ax.add_collection3d(poly)
    ax.set_axis_off()
    ax.set_facecolor("black")

    ax.view_init(elev=elev, azim=azim)
    # Set equal aspect manually
    max_range = (coords.max(axis=0) - coords.min(axis=0)).max() / 2
    mid = (coords.max(axis=0) + coords.min(axis=0)) / 2
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)


# ─── High-level plot function ─────────────────────────────────────────────────

VIEWS = {
    # (label, hemisphere, elev, azim)
    "LH lateral" : ("lh", 10,  -90),
    "LH medial"  : ("lh", 10,   90),
    "RH lateral" : ("rh", 10,   90),
    "RH medial"  : ("rh", 10,  -90),
}


def plot_atlas(atlas_name, atlas_mat_path, space,
               lh_coords, lh_faces, rh_coords, rh_faces,
               lh_labels, rh_labels, n_parcels):
    """
    Create a 1×4 figure (4 views: LH lateral/medial, RH medial/lateral)
    for one atlas parcellation.
    """
    cmap_colors = make_label_cmap(n_parcels).colors  # list of RGBA tuples

    fig = plt.figure(figsize=(20, 5), facecolor="black")
    fig.suptitle(
        f"Untamed Atlas — {atlas_name}  ({space})   [{n_parcels} parcels]",
        color="white", fontsize=15, fontweight="bold", y=1.01,
    )

    view_list = [
        # (title,       coords,     faces,     labels,    elev, azim)
        ("LH lateral",  lh_coords,  lh_faces,  lh_labels,  10, 180),
        ("LH medial",   lh_coords,  lh_faces,  lh_labels,  10,   0),
        ("RH medial",   rh_coords,  rh_faces,  rh_labels,  10, 180),
        ("RH lateral",  rh_coords,  rh_faces,  rh_labels,  10,   0),
    ]

    for idx, (title, coords, faces, labels, elev, azim) in enumerate(view_list):
        ax = fig.add_subplot(1, 4, idx + 1, projection="3d")
        ax.set_facecolor("black")
        render_hemisphere(ax, coords, faces, labels, cmap_colors, elev=elev, azim=azim)
        ax.set_title(title, color="white", fontsize=11, pad=2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"Untamed_{atlas_name}_{space}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="black")
    print(f"  Saved → {out_path}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading fsaverage6 surfaces …")
    lh_coords_fs6, lh_faces_fs6, rh_coords_fs6, rh_faces_fs6 = load_fs6_surfaces("inflated")
    lh_cortex_idx, rh_cortex_idx = load_fs6_cortex_mask()
    print(f"  FS6 cortex: lh={len(lh_cortex_idx)}, rh={len(rh_cortex_idx)}")

    print("Loading fs_LR 32k surfaces …")
    lh_coords_32k, lh_faces_32k, rh_coords_32k, rh_faces_32k = load_32k_surfaces("inflated")
    lh_idx_32k, rh_idx_32k = load_32k_cortex_mask()
    print(f"  32k: lh={lh_coords_32k.shape[0]}, rh={rh_coords_32k.shape[0]}, 32k cortex lh={len(lh_idx_32k)}, rh={len(rh_idx_32k)}")

    # ── fsaverage6 ──────────────────────────────────────────────────────────
    print("\n── fsaverage6 atlases ──")
    for name, mat_path in FS6_ATLASES.items():
        print(f"  {name} …", end=" ", flush=True)
        d       = scipy.io.loadmat(mat_path)
        labels  = d["label"].flatten()
        n_parc  = int(labels.max())
        lh_lbl, rh_lbl = unpack_fs6_labels(
            labels, lh_cortex_idx, rh_cortex_idx,
            n_lh=lh_coords_fs6.shape[0], n_rh=rh_coords_fs6.shape[0]
        )
        plot_atlas(
            name, mat_path, "fsaverage6",
            lh_coords_fs6, lh_faces_fs6,
            rh_coords_fs6, rh_faces_fs6,
            lh_lbl, rh_lbl, n_parc,
        )

    # ── fs_LR 32k ───────────────────────────────────────────────────────────
    print("\n── fs_LR 32k atlases ──")
    for name, mat_path in LR32K_ATLASES.items():
        print(f"  {name} …", end=" ", flush=True)
        d       = scipy.io.loadmat(mat_path)
        labels  = d["label"].flatten()
        n_parc  = int(labels.max())
        lh_lbl, rh_lbl = unpack_32k_labels(
            labels,
            lh_idx_32k, rh_idx_32k,
            n_lh=lh_coords_32k.shape[0],
            n_rh=rh_coords_32k.shape[0],
        )
        plot_atlas(
            name, mat_path, "fsLR32k",
            lh_coords_32k, lh_faces_32k,
            rh_coords_32k, rh_faces_32k,
            lh_lbl, rh_lbl, n_parc,
        )

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
