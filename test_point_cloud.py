import open3d as o3d
import numpy as np

def generate_point_cloud():
    # 创建随机点云数据
    points = np.random.rand(1000, 3)  # 生成1000个随机3D点
    
    # 创建点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 为点云添加随机颜色
    colors = np.random.uniform(0, 1, size=(1000, 3))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 计算点云的法向量
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    
    # 可视化点云
    o3d.visualization.draw_geometries([pcd])
    
    # 保存点云（可选）
    o3d.io.write_point_cloud("random_cloud.ply", pcd)

if __name__ == "__main__":
    generate_point_cloud()