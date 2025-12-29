"""
点云修复工具：补全、去噪、去异常点
使用 PTv3 特征进行智能修复
"""

import torch
import numpy as np
import open3d as o3d
from pathlib import Path
from inference import PTv3Inference
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import gc


class PointCloudDenoiser:
    """点云去噪器"""
    
    def __init__(self, weights_path: str = None, grid_size: float = 0.02, device: str = 'cuda'):
        """
        初始化去噪器
        
        Args:
            weights_path: PTv3 权重路径
            grid_size: 体素大小
            device: 计算设备
        """
        self.inferencer = PTv3Inference(
            weights_path=weights_path,
            grid_size=grid_size,
            device=device,
            enable_flash=False,
        )
    
    def extract_features(self, coords: np.ndarray, colors: np.ndarray = None) -> torch.Tensor:
        """提取点云特征"""
        data_dict = self.inferencer.preprocess(coords, colors)
        with torch.no_grad():
            output = self.inferencer.model(data_dict)
        return output.feat.cpu()
    
    def statistical_outlier_removal(
        self,
        coords: np.ndarray,
        nb_neighbors: int = 5,
        std_ratio: float = 2.0
    ) -> np.ndarray:
        """
        统计异常点去除
        
        Args:
            coords: 点云坐标 (N, 3)
            nb_neighbors: 邻居数量
            std_ratio: 标准差倍数阈值
            
        Returns:
            inlier_mask: 内点掩码 (N,) - True 表示保留的点
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # 统计异常点去除
        cl, ind = pcd.remove_statistical_outlier(
            nb_neighbors=nb_neighbors,
            std_ratio=std_ratio
        )
        
        inlier_mask = np.zeros(len(coords), dtype=bool)
        inlier_mask[ind] = True
        
        return inlier_mask
    
    def radius_outlier_removal(
        self,
        coords: np.ndarray,
        nb_points: int = 16,
        radius: float = 0.05
    ) -> np.ndarray:
        """
        半径异常点去除
        
        Args:
            coords: 点云坐标 (N, 3)
            nb_points: 半径内最少点数
            radius: 搜索半径
            
        Returns:
            inlier_mask: 内点掩码
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # 半径异常点去除
        cl, ind = pcd.remove_radius_outlier(
            nb_points=nb_points,
            radius=radius
        )
        
        inlier_mask = np.zeros(len(coords), dtype=bool)
        inlier_mask[ind] = True
        
        return inlier_mask
 
    def feature_based_outlier_removal(
        self,
        coords: np.ndarray,
        colors: np.ndarray = None,
        k_neighbors: int = 12,
        feature_threshold: float = 0.4
    ) -> np.ndarray:
        """
        基于特征一致性的异常点去除
        逻辑：利用 PTv3 特征向量，识别即便空间接近但语义/几何特征不匹配的噪音点
        """
        print("🔍 提取深度特征...")
        # 提取并确保特征在 CPU 内存中，避免占用显存
        features = self.extract_features(coords, colors)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        
        N = len(coords)
        # 归一化特征，用于高效计算余弦相似度
        norm = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norm + 1e-8)

        print("📊 建立空间索引并分析特征流形...")
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean', n_jobs=-1)
        nbrs.fit(coords)
        
        inlier_mask = np.ones(N, dtype=bool)
        # 分块处理，防止 21万x12x64 的中间变量撑爆内存
        batch_size = 30000
        
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            _, indices = nbrs.kneighbors(coords[i:end_idx])
            
            # 获取当前块的特征与其空间邻居的特征
            batch_feat = features[i:end_idx, np.newaxis, :] # (B, 1, 64)
            neighbor_feat = features[indices[:, 1:]]       # (B, k, 64)
            
            # 计算特征相似度：(B, k)
            similarities = np.sum(batch_feat * neighbor_feat, axis=2)
            
            # 以特征相似度的均值作为“内点得分”
            # 特征越一致，说明该点越符合局部几何表面逻辑
            avg_sim = similarities.mean(axis=1)
            inlier_mask[i:end_idx] = avg_sim > feature_threshold
        
        print(f"   - 基于特征剔除了 {np.sum(~inlier_mask)} 个离群点")
        return inlier_mask

    def bilateral_filter(
        self,
        coords: np.ndarray,
        nb_neighbors: int = 20,
        sigma_d: float = 0.1,
        sigma_n: float = 0.01
    ) -> np.ndarray:
        """
        双边滤波去噪
        
        Args:
            coords: 点云坐标 (N, 3)
            nb_neighbors: 邻居数量
            sigma_d: 空间距离标准差
            sigma_n: 法向量标准差
            
        Returns:

            filtered_coords: 滤波后的坐标
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # 估计法向量
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=nb_neighbors)
        )
        
        # 双边滤波
        filtered_pcd = pcd.filter_bilateral(
            nb_neighbors=nb_neighbors,
            sigma_d=sigma_d,
            sigma_n=sigma_n
        )
        
        return np.asarray(filtered_pcd.points)
    
    def gaussian_filter(
        self,
        coords: np.ndarray,
        k_neighbors: int = 20,
        sigma: float = 0.02
    ) -> np.ndarray:
        """
        高斯滤波去噪
        
        Args:
            coords: 点云坐标 (N, 3)
            k_neighbors: K 近邻数量
            sigma: 高斯核标准差
            
        Returns:
            filtered_coords: 滤波后的坐标
        """
        # K 近邻
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='euclidean')
        nbrs.fit(coords)
        distances, indices = nbrs.kneighbors(coords)
        
        filtered_coords = np.zeros_like(coords)
        
        for i in range(len(coords)):
            neighbors = indices[i, 1:]  # 排除自己
            neighbor_dists = distances[i, 1:]
            
            # 高斯权重
            weights = np.exp(-(neighbor_dists ** 2) / (2 * sigma ** 2))
            weights = weights / weights.sum()
            
            # 加权平均
            filtered_coords[i] = np.average(coords[neighbors], axis=0, weights=weights)
        
        return filtered_coords


class PointCloudCompletion:
    """点云补全类"""
    
    def __init__(self, weights_path: str = None, grid_size: float = 0.02, device: str = 'cuda'):
        """
        初始化补全器
        
        Args:
            weights_path: PTv3 权重路径
            grid_size: 体素大小
            device: 计算设备
        """
        self.inferencer = PTv3Inference(
            weights_path=weights_path,
            grid_size=grid_size,
            device=device,
            enable_flash=False,
        )
    
    def extract_features(self, coords: np.ndarray, colors: np.ndarray = None) -> torch.Tensor:
        """提取点云特征"""
        data_dict = self.inferencer.preprocess(coords, colors)
        with torch.no_grad():
            output = self.inferencer.model(data_dict)
        return output.feat.cpu()
    
    def poisson_surface_reconstruction(
        self,
        coords: np.ndarray,
        depth: int = 9,
        width: int = 0,
        scale: float = 1.1,
        linear_fit: bool = False
    ) -> tuple:
        """
        Poisson 表面重建
        
        Args:
            coords: 点云坐标 (N, 3)
            depth: 重建深度
            width: 宽度参数
            scale: 缩放因子
            linear_fit: 是否线性拟合
            
        Returns:
            mesh: 重建的网格
            density: 密度值
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # 估计法向量
        pcd.estimate_normals()
        
        # Poisson 重建
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
        )
        
        return mesh, densities
    
    def alpha_shape_reconstruction(
        self,
        coords: np.ndarray,
        alpha: float = 0.03
    ) -> o3d.geometry.TriangleMesh:
        """
        Alpha Shape 表面重建
        
        Args:
            coords: 点云坐标 (N, 3)
            alpha: Alpha 参数
            
        Returns:
            mesh: 重建的网格
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coords)
        
        # Alpha Shape 重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=alpha
        )
        
        return mesh
    
    def sample_points_from_mesh(
        self,
        mesh: o3d.geometry.TriangleMesh,
        num_points: int = None
    ) -> np.ndarray:
        """
        从网格采样点
        
        Args:
            mesh: 三角网格
            num_points: 采样点数（None 则使用原始点云数量）
            
        Returns:
            sampled_coords: 采样后的点坐标
        """
        if num_points is None:
            # 估算点数（根据面积）
            area = mesh.get_surface_area()
            num_points = int(area * 10000)  # 每单位面积 10000 点
        
        # 从网格采样点
        sampled_pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        return np.asarray(sampled_pcd.points)
    
    def feature_based_completion(
        self,
        coords: np.ndarray,
        colors: np.ndarray = None,
        completion_ratio: float = 0.1,
        k_neighbors: int = 15,
        inference_batch_size: int = 32768
    ) -> np.ndarray:
        """
        基于特征引导的表面补全
        逻辑：利用特征相似度加权插值，确保新生成的点落在特征最匹配的几何表面上
        """
        N = len(coords)
        # --- 阶段 1: 显存安全地提取特征 ---
        all_features = []
        with torch.no_grad():
            for i in range(0, N, inference_batch_size):
                end_idx = min(i + inference_batch_size, N)
                batch_feat = self.extract_features(coords[i:end_idx], 
                                                colors[i:end_idx] if colors is not None else None)
                all_features.append(batch_feat.cpu().numpy())
                torch.cuda.empty_cache()
        features = np.concatenate(all_features, axis=0)

        # --- 阶段 2: 识别特征复杂区（空洞边缘） ---
        nbrs = NearestNeighbors(n_neighbors=k_neighbors, n_jobs=-1).fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        # 计算特征多样性：方差越大，说明该处越可能是缺失或复杂的边缘
        feat_vars = features[indices].var(axis=1).mean(axis=1)
        
        num_new = int(N * completion_ratio)
        candidate_indices = np.argsort(feat_vars)[-num_new:]
        
        # --- 阶段 3: 特征加权生成新点 ---
        print(f"✨ 正在基于特征权重生成 {num_new} 个补全点...")
        cand_coords = coords[candidate_indices]
        cand_neighbors_idx = indices[candidate_indices]
        
        new_points = []
        for i in range(num_new):
            local_coords = coords[cand_neighbors_idx[i]]
            local_feats = features[cand_neighbors_idx[i]]
            center_feat = features[candidate_indices[i]]
            
            # 计算特征空间距离
            feat_dist = np.linalg.norm(local_feats - center_feat, axis=1)
            # 特征越相似的点权重越高（高斯核函数）
            weights = np.exp(-feat_dist / (feat_dist.mean() + 1e-8))
            weights /= weights.sum()
            
            # 基于特征权重进行位置内插
            # 这会使新生成的点向特征语义更一致的方向靠拢
            new_pos = np.sum(local_coords * weights[:, np.newaxis], axis=0)
            
            # 加入极小的扰动避免重叠
            avg_spacing = np.mean(np.linalg.norm(local_coords - cand_coords[i], axis=1))
            new_pos += np.random.randn(3) * (avg_spacing * 0.1)
            
            new_points.append(new_pos)

        print(f"   - 补全完成")
        return np.vstack([coords, np.array(new_points, dtype=np.float32)])


def main():
    """示例：点云去噪和补全"""
    import argparse
    
    parser = argparse.ArgumentParser(description='点云修复：去噪、去异常点、补全')
    parser.add_argument('--input', type=str, default="pointcloud/mouse.pcd", help='输入点云路径')
    parser.add_argument('--output', type=str, default="pointcloud/mouse_outputcloud.pcd", help='输出点云路径')
    parser.add_argument('--weights', type=str, default='models/sonata_small.pth', help='PTv3 权重路径')
    parser.add_argument('--mode', type=str, choices=['denoise', 'outlier', 'completion', 'all'],
                        default='all', help='处理模式')
    parser.add_argument('--visualize', action='store_true', help='可视化结果')
    
    args = parser.parse_args()
    
    # 读取点云
    print("📄 读取点云...")
    pcd = o3d.io.read_point_cloud(args.input)
    coords = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"   - 原始点数: {len(coords)}")
    
    processed_coords = coords.copy()
    processed_colors = colors.copy() if colors is not None else None
    
    if args.mode in ['denoise', 'all']:
        print("\n🧹 去噪处理...")
        denoiser = PointCloudDenoiser(weights_path=args.weights)
        
        # 统计异常点去除
        print("  1️⃣ 统计异常点去除...")
        inlier_mask = denoiser.statistical_outlier_removal(processed_coords)
        processed_coords = processed_coords[inlier_mask]
        if processed_colors is not None:
            processed_colors = processed_colors[inlier_mask]
        print(f"     - 剩余点数: {len(processed_coords)}")
        
        # 基于特征的异常点去除
        if args.mode == 'all':
            print("  2️⃣ 基于特征的异常点去除...")
            try:
                inlier_mask = denoiser.feature_based_outlier_removal(
                    processed_coords, processed_colors
                )
                processed_coords = processed_coords[inlier_mask]
                if processed_colors is not None:
                    processed_colors = processed_colors[inlier_mask]
                print(f"     - 剩余点数: {len(processed_coords)}")
            except Exception as e:
                print(f"     - ⚠️ 特征异常点去除失败: {e}")
        
        # 双边滤波
        print("  3️⃣ 双边滤波...")
        try:
            processed_coords = denoiser.bilateral_filter(processed_coords)
            print(f"     - 滤波完成")
        except Exception as e:
            print(f"     - ⚠️ 双边滤波失败: {e}")
    
    if args.mode in ['completion', 'all']:
        print("\n✨ 点云补全...")
        completer = PointCloudCompletion(weights_path=args.weights)
        
        try:
            # 基于特征的补全
            processed_coords = completer.feature_based_completion(
                processed_coords, processed_colors, completion_ratio=0.1
            )
        except Exception as e:
            print(f"   - ⚠️ 特征补全失败: {e}")
    
    # 保存结果
    print(f"\n💾 保存结果...")
    output_pcd = o3d.geometry.PointCloud()
    output_pcd.points = o3d.utility.Vector3dVector(processed_coords)
    if processed_colors is not None:
        output_pcd.colors = o3d.utility.Vector3dVector(processed_colors)
    o3d.io.write_point_cloud(args.output, output_pcd)
    print(f"   - 输出点数: {len(processed_coords)}")
    print(f"   - 已保存: {args.output}")
    
    # 可视化
    if args.visualize:
        print("\n👁️ 可视化结果...")
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(coords)
        if colors is not None:
            original_pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            original_pcd.paint_uniform_color([1, 0, 0])  # 红色
        
        output_pcd.paint_uniform_color([0, 1, 0])  # 绿色
        o3d.visualization.draw_geometries([output_pcd])
   
if __name__ == '__main__':
    main()

