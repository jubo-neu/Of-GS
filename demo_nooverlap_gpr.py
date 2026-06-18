import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import trimesh

import open3d as o3d
import gtsam
from gtsam import Pose3, Rot3, noiseModel, NonlinearFactorGraph, Values, LevenbergMarquardtParams, \
    LevenbergMarquardtOptimizer
from vggt.models.vggt import VGGT

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

print(f"Using device: {device}")
print(f"Using dtype: {dtype}")

model = VGGT()
model_weight_path = ""
model.load_state_dict(torch.load(model_weight_path, map_location=device))
model = model.to(device)
model.eval()


image_names = []


def load_and_preprocess_images(image_paths, resolution=518):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images_list = []
    for path in image_paths:
        img = Image.open(path).convert('RGB')
        img = transform(img)  # [3, H, W]
        images_list.append(img)
    images = torch.stack(images_list, dim=0)  # [N, 3, H, W]
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)
    return images


images = load_and_preprocess_images(image_names).to(device)
N = len(image_names)
H, W = 518, 518


def pose_encoding_to_extri_intri(pose_enc, resolution=518):
    """
    pose_enc: [N, 9]
    """
    N = pose_enc.shape[0]
    fx = fy = resolution
    cx = cy = resolution / 2.0

    K = torch.eye(3, device=pose_enc.device).repeat(N, 1, 1)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy

    rot_6d = pose_enc[:, :6]  # [N, 6]
    tvec = pose_enc[:, 6:]  # [N, 3]

    def rotation_6d_to_matrix(d6):
        a1, a2 = d6[:, 0:3], d6[:, 3:6]
        b1 = F.normalize(a1, dim=1)
        b2 = a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=2)

    R = rotation_6d_to_matrix(rot_6d)  # [N, 3, 3]
    T = tvec  # [N, 3]

    extrinsic = torch.eye(4, device=pose_enc.device).repeat(N, 1, 1)
    extrinsic[:, :3, :3] = R
    extrinsic[:, :3, 3] = T

    return extrinsic, K


@torch.no_grad()
def run_vggt(model, images, dtype):
    B, C, H, W = images.shape
    assert C == 3

    with torch.cuda.amp.autocast(dtype=dtype):
        batched_images = images[None]  # [1, B, 3, H, W]
        aggregated_tokens_list, ps_idx = model.aggregator(batched_images)

        print(f"aggregated_tokens_list length: {len(aggregated_tokens_list)}")
        for i, t in enumerate(aggregated_tokens_list):
            print(f"  Layer {i}: {t.shape}")

        pose_enc_list = model.camera_head(aggregated_tokens_list)
        pose_enc = pose_enc_list[-1]  # [1, B, 9]

        pose_enc = pose_enc.squeeze(0)  # [B, 9]
        print(f"🔍 pose_enc after squeeze: {pose_enc.shape}")

        if pose_enc.shape[0] == 0:
            raise ValueError("pose_enc is empty!")

        extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc, resolution=H)

        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, batched_images, ps_idx)

        print(f"depth_map shape: {depth_map.shape}")
        print(f"depth_conf shape: {depth_conf.shape}")

    # 转为 numpy
    extrinsics = extrinsics.cpu().numpy()  # (N, 4, 4)
    intrinsics = intrinsics.cpu().numpy()  # (N, 3, 3)
    depth_maps = depth_map.squeeze(0).cpu().numpy()  # -> [N, H, W]
    depth_confs = depth_conf.squeeze(0).cpu().numpy()

    if depth_maps.ndim == 4 and depth_maps.shape[-1] == 1:
        depth_maps = depth_maps.squeeze(-1)  # -> [N, H, W]

    if depth_confs.ndim == 4 and depth_confs.shape[-1] == 1:
        depth_confs = depth_confs.squeeze(-1)

    return extrinsics, intrinsics, depth_maps, depth_confs


print("Starting VGGT inference...")
extrinsics, intrinsics, depth_maps, depth_confs = run_vggt(model, images, dtype)
print("Inference completed!")

print(f"Extrinsics shape: {extrinsics.shape}")
print(f"Intrinsics shape: {intrinsics.shape}")
print(f"Depth maps shape: {depth_maps.shape}")

local_point_clouds = []
world_point_clouds_coarse = []
rgb_images = []

xx, yy = np.meshgrid(np.arange(W), np.arange(H))
pixel_coords = np.stack([xx, yy, np.ones_like(xx)], axis=-1)  # (H, W, 3)

for path in image_names:
    img = Image.open(path).convert('RGB')
    img_resized = img.resize((W, H), Image.LANCZOS)
    rgb_images.append(np.array(img_resized))  # (H, W, 3)

for i in range(N):
    depth = depth_maps[i]  # (H, W)
    K = intrinsics[i]  # (3, 3)
    T_c2w = extrinsics[i]  # (4, 4)
    rgb = rgb_images[i]  # (H, W, 3)

    mask = (depth > 0.1) & (depth < 100) & np.isfinite(depth)  # (H, W)
    if not np.any(mask):
        print(f"No valid depth for image {i}")
        continue

    yy_idx, xx_idx = np.where(mask)  # (P,), (P,)
    pixels = np.stack([xx_idx, yy_idx, np.ones_like(xx_idx)], axis=1)  # (P, 3)
    valid_depth = depth[mask]  # (P,)

    cam_points = (np.linalg.inv(K) @ pixels.T) * valid_depth  # (3, P)
    cam_points = cam_points.T  # (P, 3)

    cam_points_hom = np.concatenate([cam_points, np.ones((len(cam_points), 1))], axis=1)  # (P, 4)
    world_points_hom = (T_c2w @ cam_points_hom.T).T  # (P, 4)
    world_points = world_points_hom[:, :3]  # (P, 3)

    colors = rgb[xx_idx, yy_idx]  # (P, 3)

    local_point_clouds.append((cam_points, colors))
    world_point_clouds_coarse.append((world_points, colors))

print(f"Generated {len(local_point_clouds)} local point clouds for ICP.")

def compute_icp_transform(source_points, target_points, max_correspondence_distance=0.05):
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    source_pcd = source_pcd.voxel_down_sample(voxel_size=0.02)
    target_pcd = target_pcd.voxel_down_sample(voxel_size=0.02)

    result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
    )
    return result.transformation  # (4, 4)


relative_poses = []

print("Running pairwise ICP...")
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        src_points, _ = local_point_clouds[i]
        tgt_points, _ = local_point_clouds[j]
        T_icp = compute_icp_transform(src_points, tgt_points, max_correspondence_distance=0.1)
        relative_poses.append((i, j, T_icp))
        print(f"  ICP {i} -> {j}: success={not np.isnan(T_icp).any()}")

print(f"Collected {len(relative_poses)} relative pose constraints.")


def global_pose_refinement(initial_poses, relative_poses, noise_model_prior=None, noise_model_meas=None):
    """
    initial_poses: [N, 4, 4]  (T_c2w)
    relative_poses: list of (i, j, T_i_to_j)
    """
    graph = NonlinearFactorGraph()
    initial_estimate = Values()

    if noise_model_prior is None:
        noise_model_prior = noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]))
    if noise_model_meas is None:
        noise_model_meas = noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]))

    for i, T in enumerate(initial_poses):
        pose3 = Pose3(Rot3(T[:3, :3]), T[:3, 3])
        initial_estimate.insert(i, pose3)
        if i == 0:
            graph.add(gtsam.PriorFactorPose3(i, pose3, noise_model_prior))

    for i, j, T_ij in relative_poses:
        T_ij_gtsam = Pose3(Rot3(T_ij[:3, :3]), T_ij[:3, 3])
        graph.add(gtsam.BetweenFactorPose3(i, j, T_ij_gtsam, noise_model_meas))

    params = LevenbergMarquardtParams()
    optimizer = LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    refined_poses = []
    for i in range(len(initial_poses)):
        pose3_opt = result.atPose3(i)
        T = np.eye(4)
        T[:3, :3] = pose3_opt.rotation().matrix()
        T[:3, 3] = pose3_opt.translation()
        refined_poses.append(T)

    return np.array(refined_poses)


print("Starting Global Pose Refinement (GPR)...")
refined_extrinsics = global_pose_refinement(extrinsics, relative_poses)
print("GPR completed!")

all_points_world_refined = []
all_colors_refined = []

for i in range(N):
    depth = depth_maps[i]
    K = intrinsics[i]
    T_c2w = refined_extrinsics[i]
    rgb = rgb_images[i]

    mask = (depth > 0.1) & (depth < 100) & np.isfinite(depth)
    if not np.any(mask):
        continue

    yy_idx, xx_idx = np.where(mask)
    pixels = np.stack([xx_idx, yy_idx, np.ones_like(xx_idx)], axis=1)
    valid_depth = depth[mask]

    cam_points = (np.linalg.inv(K) @ pixels.T) * valid_depth
    cam_points = cam_points.T

    cam_points_hom = np.concatenate([cam_points, np.ones((len(cam_points), 1))], axis=1)
    world_points_hom = (T_c2w @ cam_points_hom.T).T
    world_points = world_points_hom[:, :3]

    colors = rgb[xx_idx, yy_idx]

    all_points_world_refined.append(world_points)
    all_colors_refined.append(colors)

merged_points_refined = np.vstack(all_points_world_refined)
merged_colors_refined = np.vstack(all_colors_refined)

print(f"Final refined point cloud: {merged_points_refined.shape[0]} points")

output_dir = "output_predictions/point_clouds"
os.makedirs(output_dir, exist_ok=True)

ply_coarse = os.path.join(output_dir, "merged_pointcloud_coarse.ply")
coarse_cloud = trimesh.PointCloud(np.vstack([p for p, _ in world_point_clouds_coarse]),
                                  colors=np.vstack([c for _, c in world_point_clouds_coarse]))
coarse_cloud.export(ply_coarse)
print(f"Saved coarse point cloud to {ply_coarse}")

ply_refined = os.path.join(output_dir, "merged_pointcloud_gpr.ply")
refined_cloud = trimesh.PointCloud(merged_points_refined, colors=merged_colors_refined)
refined_cloud.export(ply_refined)
print(f"Saved refined point cloud (GPR) to {ply_refined}")

print("All done! You now have both coarse and GPR-refined point clouds.")
