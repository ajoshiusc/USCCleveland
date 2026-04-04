# Untamed Atlases Toolkit

This repository contains tools to process, convert, and visualize the Untamed atlas parcellations for different neuroimaging platforms.

## Files
- `create_brainstorm_atlases.py`: Converts the Untamed parcellations into Brainstorm Scout (`.mat`) files. The generated scouts can be individually loaded into native Brainstorm anatomy surfaces.
- `create_brainstorm_default_anatomy.py`: Provides tools to generate and compile fully standalone "Default Anatomy" Brainstorm directories for `fsaverage6` and `fsLR32k` spaces, natively embedding the Untamed atlas parcellations.
- `create_brainsuite_templates.py`: Converts the Untamed parcellations into templates natively compatible with BrainSuite (SVReg), outputting FreeSurfer-style `.annot` files for `fsaverage6` space and `.dlabel.nii` files for `fs_LR 32k` space.
- `plot_untamed_atlas_pyvista.py`: An interactive, PyVista-based 3D visualization tool for exploring the Untamed atlases on cortical surfaces. Note: This replaces an older `matplotlib`-based script.
