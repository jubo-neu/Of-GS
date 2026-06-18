# Of-GS
## [Paper] Of-GS: Overlap-free Gaussian Splatting via Visual Geometry Grounded Transformer for Few-shot Novel View Synthesis
These are the official code releases.
- Graphical Abstract:
<img src="https://github.com/jubo-neu/Of-GS/blob/main/GA.png" alt="Teaser Image">

## Highlights
- Enables novel view synthesis from only three overlap-free input images
- Introduces a closed-loop Global Pose Refinement block for geometric consistency
- Fuses multi-view point clouds via point-to-plane ICP alignment without SfM
- Proposes Fusion-point Gaussian Initialization module to stabilize 3DGS training
- Achieves state-of-the-art quality without relying on COLMAP or pose priors

## TODO
- [x] Benchmarks
- [x] Models and Scripts

## Preparation for training
1. To start, we prefer creating the environment using pip:
```
pip install -r requirements.txt
```
2. Getting the data
- Download the complete [UrbanScene3D](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip).
- Download the complete [LLFF](https://s3.eu-central-1.amazonaws.com/avg-projects/differentiable_volumetric_rendering/data/NMR_Dataset.zip).
- Download the complete [Tanks and Temples](https://www.tanksandtemples.org/).
- Due to the double-blind policy, our self-collected dataset will be made public with the OneDrive link after the paper is accepted.
