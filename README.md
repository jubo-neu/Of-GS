# Of-GS
## [Paper] Of-GS: Overlap-free Gaussian Splatting via Visual Geometry Grounded Transformer for Few-shot Novel View Synthesis
These are the official code releases.

❗️This repository currently provides the core model (.py files) of our method. The complete running process and detailed operating instructions will be available upon acceptance.
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

## Preparation for running
1. To start, we prefer creating the environment using pip:
```
pip install -r requirements.txt
```
2. Getting the data
- Download the complete [UrbanScene3D](https://vcc.tech/UrbanScene3D).
- Download the complete [LLFF](https://drive.google.com/drive/folders/1cK3UDIJqKAAm7zyrxRYVFJ0BRMgrwhh4).
- Download the complete [Tanks and Temples](https://www.tanksandtemples.org/).
- Due to the double-blind policy, our self-collected dataset will be made public with the OneDrive link after the paper is accepted.

## Running
1. Prepare images for running. Create `examples` to store input views. Create `output_predictions` to store outputs.

- Ensuring that you have the following files:
```bash
Of-GS
|-- examples
    |-- data_1
        |-- 00000.png
        |-- 00001.png
        |-- ...
    |-- data_2
        |-- 00000.png
        |-- 00001.png
        |-- ...
    |-- ...
|-- output_predictions
    |-- depth_maps
    |-- point_clouds
```
2. Download the pretrained VGGT-1B model [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt).
3. Start:
- `demo_colmap.py`: A script that runs the VGGT model to estimate camera poses and depth maps from input images, then constructs a COLMAP-compatible sparse 3D reconstruction—with optional bundle adjustment (BA) or feedforward mode—and saves the result in standard COLMAP format.
- `demo_nooverlap_gpr.py`: A script that runs VGGT to estimate initial camera poses and depth maps, then refines the global camera trajectory using pairwise ICP and global pose refinement (GPR) with GTSAM, producing both coarse and refined 3D point clouds saved as PLY files.
4. Run Gaussian point sampling and Gaussian rasterization rendering by [3DGS](https://github.com/graphdeco-inria/gaussian-splatting).

## Notice
This repository currently provides the core model (.py files) of our method. The complete running process and detailed operating instructions will be available upon acceptance.
