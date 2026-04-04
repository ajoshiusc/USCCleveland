#!/usr/bin/env python3
"""
plot_untamed_atlas_pyvista.py
----------------------------
Plot the Untamed atlas parcellations using PyVista (interactive).

It reads the standalone Brainstorm anatomies (`tess_cortex.mat`)
directly instead of loading raw external surface files.

Usage:
    source /home/ajoshi/Projects/USCCleveland/.venv/bin/activate
    python plot_untamed_atlas_pyvista.py

Author: auto-generated
"""

import os
import numpy as np
import scipy.io
import pyvista as pv

# ─── Paths ────────────────────────────────────────────────────────────────────

# Point directly to the generated brainstorm default anatomies which embed the surfaces + scouts
BST_ANATOMY_DIR = "brainstorm_default_anatomies"

FS6_TESS_PATH = os.path.join(BST_ANATOMY_DIR, "Untamed_fsaverage6", "tess_cortex.mat")
LR32K_TESS_PATH = os.path.join(BST_ANATOMY_DIR, "Untamed_fsLR32k", "tess_cortex.mat")

# ─── PyVista Helpers ─────────────────────────────────────────────────────────

def nib_to_pyvista(coords, faces):
    """Convert nibabel/matlab geometry to pyvista PolyData."""
    pv_faces = np.column_stack((np.full(len(faces), 3), faces)).ravel()
    return pv.PolyData(coords, pv_faces)

# ─── High-level plot function ─────────────────────────────────────────────────

def plot_atlas_pyvista_combined(atlas_name, space, coords, faces, point_colors):
    """
    Create an interactive 4-view plot with PyVista using a single combined mesh.
    """
    
    mesh = nib_to_pyvista(coords, faces)
    
    # We assign RGB colors per vertex
    # To avoid boundary interpolation issues, we can compute face colors 
    # instead by taking the majority color or simply using point data.
    # But since the user scout colors are identical across the face, point data is ok,
    # except at boundaries. Let's convert point colors to cell colors to keep boundaries sharp.
    
    face_v0 = point_colors[faces[:, 0]]
    face_v1 = point_colors[faces[:, 1]]
    face_v2 = point_colors[faces[:, 2]]
    
    # Majority vote per channel (since colors are shared by vertices of the same parcel)
    # At boundaries, v0, v1, v2 might differ. This simple heuristic assigns 
    # v0's color if it matches v1 or v2, else v1's color.
    match_01 = np.all(face_v0 == face_v1, axis=1)
    match_02 = np.all(face_v0 == face_v2, axis=1)
    
    cell_colors = np.where(match_01[:, None] | match_02[:, None], face_v0, face_v1)
    
    mesh.cell_data["colors"] = cell_colors
    
    plotter = pv.Plotter(shape=(1, 4), title=f"Untamed Atlas - {atlas_name} ({space})")
    plotter.set_background("black")
    
    mesh_args = dict(scalars="colors", rgb=True, smooth_shading=True, preference="cell")
    
    def _add_view(row, col, title, negative):
        plotter.subplot(row, col)
        plotter.add_text(title, position="upper_left", font_size=10, color="white")
        plotter.add_mesh(mesh, **mesh_args)
        plotter.reset_camera()
        plotter.view_yz(negative=negative)
        plotter.reset_camera_clipping_range()

    # The mesh is combined. The left hemisphere is at negative X usually? 
    # For fsaverage6 we artificially shifted them. For 32k they are naturally separated.
    # "negative=True" views from the left side (LH Lateral or RH Medial).
    # Since they are separated, from Left side you see LH Lateral and RH Medial through the gap.
    # From Right side you see RH Lateral and LH Medial through the gap.
    
    _add_view(0, 0, "View Left", True)
    _add_view(0, 1, "View Right", False)
    
    # Top and Bottom views for the remaining subplots
    plotter.subplot(0, 2)
    plotter.add_text("View Top", position="upper_left", font_size=10, color="white")
    plotter.add_mesh(mesh, **mesh_args)
    plotter.reset_camera()
    plotter.view_xy(negative=False) # View from +Z
    plotter.reset_camera_clipping_range()
    
    plotter.subplot(0, 3)
    plotter.add_text("View Bottom", position="upper_left", font_size=10, color="white")
    plotter.add_mesh(mesh, **mesh_args)
    plotter.reset_camera()
    plotter.view_xy(negative=True) # View from -Z
    plotter.reset_camera_clipping_range()
    
    plotter.link_views()
    print(f"Showing interactive plot for {atlas_name} ({space}) ...")
    plotter.show()

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Which space would you like to plot?")
    print("  1. fsaverage6")
    print("  2. fs_LR 32k")
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        mat_path = FS6_TESS_PATH
        space = "fsaverage6"
    elif choice == "2":
        mat_path = LR32K_TESS_PATH
        space = "fsLR32k"
    else:
        print("Invalid choice. Exiting.")
        return

    if not os.path.exists(mat_path):
        print(f"Error: {mat_path} does not exist.")
        print("Please run create_brainstorm_default_anatomy.py first.")
        return

    print(f"Loading {space} from {mat_path} …")
    mat = scipy.io.loadmat(mat_path, simplify_cells=True)
    
    vertices = mat['Vertices']
    faces = np.array(mat['Faces'], dtype=int) - 1 # 1-indexed to 0-indexed
    
    atlases = mat['Atlas']
    # If there is only one atlas, scipy might not load it as a list
    if not isinstance(atlases, (list, np.ndarray)):
        atlases = [atlases]
        
    print(f"\nAvailable {space} atlases:")
    for i, a in enumerate(atlases):
        print(f"  {i+1}. {a['Name']}")
    
    atlas_choice = int(input(f"Pick an atlas (1-{len(atlases)}): ")) - 1
    chosen_atlas = atlases[atlas_choice]
    
    print(f"Loading colors for {chosen_atlas['Name']} …")
    
    point_colors = np.zeros((len(vertices), 3))
    # Default to a dark gray for medial wall / unlabeled
    point_colors[:] = [0.15, 0.15, 0.15]
    
    scouts = chosen_atlas['Scouts']
    # If there's only one scout, make it a list
    if not isinstance(scouts, (list, np.ndarray)):
        scouts = [scouts]
        
    for scout in scouts:
        if 'Vertices' in scout:
            # Check if Vertices is empty string or empty array
            if isinstance(scout['Vertices'], str) and scout['Vertices'] == '[]':
                continue
            
            # Substract 1 to adapt from 1-indexed Brainstorm vertices to 0-indexed
            try:
                verts = np.array(scout['Vertices'], dtype=int) - 1
                point_colors[verts] = scout['Color']
            except TypeError:
                # If only a single vertex
                v = int(scout['Vertices']) - 1
                point_colors[v] = scout['Color']
                
    plot_atlas_pyvista_combined(
        chosen_atlas['Name'], 
        space,
        vertices, 
        faces, 
        point_colors
    )

if __name__ == "__main__":
    main()
