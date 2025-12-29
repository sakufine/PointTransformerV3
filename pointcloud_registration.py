"""
点云配准（对齐两个点云）
使用 PTv3 特征进行特征匹配 + ICP 精确配准
"""

import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from inference import PTv3Inference
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree

class PointCloudRegistration:
    """点云配准类"""
    
    def __init__(self, weights_path: str = None, grid_size: float = 0.02, device: str = 'cuda'):
        """
        初始化配准器
        
        Args:
            weights_path: PTv3 权重路径
            grid_size: 体素大小
            device: 计算设备
        """
        self.inferencer = PTv3Inference(
            weights_path=weights_path,
            grid_size=grid_size,
            device=device,
            enable_flash=True,  # 可以启用 Flash Attention
        )
    
    def extract_features(self, coords: np.ndarray, colors: np.ndarray = None) -> torch.Tensor:
        data_dict = self.inferencer.preprocess(coords, colors)
        with torch.no_grad():
            output = self.inferencer.model(data_dict)
        feat = output.feat.cpu()
        # 新增：L2 归一化
        feat = torch.nn.functional.normalize(feat, p=2, dim=1)
        return feat
    
    def feature_matching(
        self, 
        feat1: torch.Tensor, 
        feat2: torch.Tensor,
        ratio_threshold: float = 0.8
    ) -> tuple:
        """
        使用 KDTree 优化特征匹配，避免 53GB 的内存溢出
        """
        # 转换为 float32 节省内存
        feat1_np = feat1.numpy().astype(np.float32)
        feat2_np = feat2.numpy().astype(np.float32)
        
        # 构建目标点云特征的 KDTree
        tree = cKDTree(feat2_np)
        
        # 查询 feat1 中每个点在 feat2 中的 2 个最近邻 (用于 Ratio Test)
        # k=2, workers=-1 表示使用所有 CPU 核心并行加速
        distances, indices = tree.query(feat1_np, k=2, workers=-1)
        
        matches = []
        match_distances = []
        
        # 执行 Lowe's Ratio Test
        for i in range(len(feat1_np)):
            d0, d1 = distances[i]
            if d0 < ratio_threshold * d1:
                matches.append([i, indices[i, 0]])
                match_distances.append(d0)
                
        if len(matches) == 0:
            return np.array([]), np.array([])
        
        matches = np.array(matches)
        match_distances = np.array(match_distances)
        
        # 按照距离排序
        sort_idx = np.argsort(match_distances)
        return matches[sort_idx], match_distances[sort_idx]
    
    def ransac_alignment(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        matches: np.ndarray,
        num_iterations: int = 5000,
        threshold: float = 0.5,
        num_samples: int = 4
    ) -> tuple:
        """
        RANSAC 粗配准
        
        Args:
            coords1: 点云1坐标 (N1, 3)
            coords2: 点云2坐标 (N2, 3)
            matches: 匹配点对 (M, 2)
            num_iterations: RANSAC 迭代次数
            threshold: 内点阈值
            num_samples: 每次采样点数
            
        Returns:
            transform: 4x4 变换矩阵
            inliers: 内点索引
        """
        if len(matches) < num_samples:
            return np.eye(4), np.array([])
        
        best_transform = np.eye(4)
        best_inliers = np.array([])
        max_inliers = 0
        
        points1 = coords1[matches[:, 0]]
        points2 = coords2[matches[:, 1]]
        
        for _ in range(num_iterations):
            # 随机采样
            sample_idx = np.random.choice(len(matches), num_samples, replace=False)
            sample_p1 = points1[sample_idx]
            sample_p2 = points2[sample_idx]
            
            # 计算变换矩阵（最小二乘）
            try:
                # 计算中心
                center1 = sample_p1.mean(axis=0)
                center2 = sample_p2.mean(axis=0)
                
                # 去中心化
                p1_centered = sample_p1 - center1
                p2_centered = sample_p2 - center2
                
                # SVD 分解求旋转
                H = p1_centered.T @ p2_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                
                # 确保是旋转矩阵（行列式为1）
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                
                # 计算平移
                t = center2 - R @ center1
                
                # 构建变换矩阵
                transform = np.eye(4)
                transform[:3, :3] = R
                transform[:3, 3] = t
                
                # 应用变换并计算内点
                transformed_p1 = (R @ points1.T).T + t
                distances = np.linalg.norm(transformed_p1 - points2, axis=1)
                inliers = np.where(distances < threshold)[0]
                
                if len(inliers) > max_inliers:
                    max_inliers = len(inliers)
                    best_transform = transform
                    best_inliers = matches[inliers]
                    
            except:
                continue
        
        return best_transform, best_inliers
    
    def icp_refinement(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        init_transform: np.ndarray = None,
        max_iterations: int = 50,
        threshold: float = 0.02
    ) -> tuple:
        """
        ICP 精配准
        
        Args:
            coords1: 点云1坐标 (N1, 3)
            coords2: 点云2坐标 (N2, 3)
            init_transform: 初始变换矩阵 (4x4)
            max_iterations: 最大迭代次数
            threshold: 距离阈值
            
        Returns:
            transform: 最终变换矩阵 (4x4)
            fitness: 配准质量分数 [0, 1]
        """
        # 转换为 Open3D 格式
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(coords1)
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(coords2)
        
        # 初始变换
        if init_transform is not None:
            pcd1.transform(init_transform)
        
        # ICP 配准
        result = o3d.pipelines.registration.registration_icp(
            pcd1, pcd2, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        )
        
        # 组合变换
        if init_transform is not None:
            final_transform = result.transformation @ init_transform
        else:
            final_transform = result.transformation
        
        return final_transform, result.fitness
    
    def register(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        colors1: np.ndarray = None,
        colors2: np.ndarray = None,
        use_feature_matching: bool = True
    ) -> tuple:
        """
        完整配准流程
        
        Args:
            coords1: 源点云坐标 (N1, 3)
            coords2: 目标点云坐标 (N2, 3)
            colors1: 源点云颜色，可选
            colors2: 目标点云颜色，可选
            use_feature_matching: 是否使用特征匹配
            
        Returns:
            transform: 变换矩阵 (4x4)，将 coords1 变换到 coords2 的坐标系
            aligned_coords1: 对齐后的源点云坐标
            fitness: 配准质量分数
        """
        print("🔄 开始点云配准...")
        
        if use_feature_matching:
            # 方法1：特征匹配 + RANSAC + ICP
            print("  1️⃣ 提取特征...")
            feat1 = self.extract_features(coords1, colors1)
            feat2 = self.extract_features(coords2, colors2)
            print(f"     - 点云1特征: {feat1.shape}")
            print(f"     - 点云2特征: {feat2.shape}")
            
            print("  2️⃣ 特征匹配...")
            matches, match_distances = self.feature_matching(feat1, feat2)
            print(f"     - 找到 {len(matches)} 个匹配点对")
            
            if len(matches) > 3:
                print("  3️⃣ RANSAC 粗配准...")
                init_transform, inliers = self.ransac_alignment(
                    coords1, coords2, matches
                )
                print(f"     - 内点数量: {len(inliers)}")
            else:
                print("  ⚠️ 匹配点对太少，使用单位矩阵作为初始变换")
                init_transform = np.eye(4)
        else:
            # 方法2：直接 ICP（需要初始位置接近）
            init_transform = np.eye(4)
        
        print("  4️⃣ ICP 精配准...")
        final_transform, fitness = self.icp_refinement(
            coords1, coords2, init_transform
        )
        print(f"✅ 配准完成! 质量分数: {fitness:.3f}")
        
        # 应用变换
        R = final_transform[:3, :3]
        t = final_transform[:3, 3]
        aligned_coords1 = (R @ coords1.T).T + t
        
        return final_transform, aligned_coords1, fitness
    
    def visualize_registration(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        aligned_coords1: np.ndarray,
        colors1: np.ndarray = None,
        colors2: np.ndarray = None
    ):
        """
        可视化配准结果
        
        Args:
            coords1: 原始源点云
            coords2: 目标点云
            aligned_coords1: 对齐后的源点云
            colors1: 源点云颜色
            colors2: 目标点云颜色
        """
        # 创建点云
        pcd1_orig = o3d.geometry.PointCloud()
        pcd1_orig.points = o3d.utility.Vector3dVector(coords1)
        if colors1 is not None:
            pcd1_orig.colors = o3d.utility.Vector3dVector(colors1)
        else:
            pcd1_orig.paint_uniform_color([1, 0, 0])  # 红色
        
        pcd1_aligned = o3d.geometry.PointCloud()
        pcd1_aligned.points = o3d.utility.Vector3dVector(aligned_coords1)
        if colors1 is not None:
            pcd1_aligned.colors = o3d.utility.Vector3dVector(colors1)
        else:
            pcd1_aligned.paint_uniform_color([0, 1, 0])  # 绿色
        
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(coords2)
        if colors2 is not None:
            pcd2.colors = o3d.utility.Vector3dVector(colors2)
        else:
            pcd2.paint_uniform_color([0, 0, 1])  # 蓝色
        
        # 可视化
        o3d.visualization.draw_geometries([pcd1_aligned, pcd2])


def main():
    """示例：配准两个点云"""
    import argparse
    
    parser = argparse.ArgumentParser(description='点云配准')
    parser.add_argument('--source', type=str, default="pointcloud/mouse.pcd", help='源点云路径')
    parser.add_argument('--target', type=str, default="pointcloud/mouse_right.pcd", help='目标点云路径')
    parser.add_argument('--weights', type=str, default="models/sonata_small.pth", help='PTv3 权重路径')
    parser.add_argument('--output', type=str, default="pointcloud/mouse_registered.pcd", help='输出对齐后的点云路径')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    
    args = parser.parse_args()
    
    # 读取点云
    print("📄 读取点云...")
    pcd1 = o3d.io.read_point_cloud(args.source)
    pcd2 = o3d.io.read_point_cloud(args.target)
    
    coords1 = np.asarray(pcd1.points)
    coords2 = np.asarray(pcd2.points)
    colors1 = np.asarray(pcd1.colors) if pcd1.has_colors() else None
    colors2 = np.asarray(pcd2.colors) if pcd2.has_colors() else None
    
    print(f"   - 源点云: {len(coords1)} 个点")
    print(f"   - 目标点云: {len(coords2)} 个点")
    
    # 创建配准器
    registrar = PointCloudRegistration(weights_path=args.weights)
    
    # 配准
    transform, aligned_coords1, fitness = registrar.register(
        coords1, coords2, colors1, colors2
    )
    
    # 保存结果
    if args.output:
        aligned_pcd = o3d.geometry.PointCloud()
        aligned_pcd.points = o3d.utility.Vector3dVector(aligned_coords1)
        if colors1 is not None:
            aligned_pcd.colors = o3d.utility.Vector3dVector(colors1)
        o3d.io.write_point_cloud(args.output, aligned_pcd)
        print(f"💾 对齐后的点云已保存: {args.output}")
    
    # 可视化
    if args.visualize:
        registrar.visualize_registration(
            coords1, coords2, aligned_coords1, colors1, colors2
        )
    
    return transform, aligned_coords1, fitness


if __name__ == '__main__':
    main()

