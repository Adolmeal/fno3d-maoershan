fno3d-maoershan: 林区风场 FNO/Tranformer的混合神经网络替代pimpleFoam求解的训练模型

该项目旨在构建一个基于 Fourier Neural Operator (FNO) 的 3D 神经网络模型，用于模拟和预测帽儿山林区复杂地形中的风场分布。

📌 项目目标

构建高分辨率风场模拟数据集（基于 10m DEM + ERA5 风速 + NDVI 植被信息）

训练一个可替代 OpenFOAM/Fluent 的高精度风场神经网络模型

支持与 Fluent 实时耦合，提升大规模 CFD 模拟效率

🚀 快速开始

# 克隆仓库
https://github.com/Adolmeal/fno3d-maoershan.git

# 创建虚拟环境
python -m venv fno3d_env
source fno3d_env/bin/activate

# 安装依赖
pip install -r requirements.txt

# 查看处理后数据可视化（可选）
python scripts/visualize_fields.py

python visualize_feature_maps.py

![feature_map_1](https://github.com/user-attachments/assets/83832079-263d-4238-93de-af612bf4fcae)

![feature_map_2](https://github.com/user-attachments/assets/54365b35-17e9-4f3e-879b-2bc1e546ce96)

![feature_map_1_smooth](https://github.com/user-attachments/assets/95f6a0ad-0193-4494-9d9e-41b0ec2cf8ac)

![feature_map_2_smooth](https://github.com/user-attachments/assets/9f66ea43-9d38-4d18-a7c5-b66860120d56)

![feature_map_3](https://github.com/user-attachments/assets/b973d65c-d2e1-4661-87e9-e43424b9393c)
![feature_map_4](https://github.com/user-attachments/assets/26fc689f-ac40-4568-923b-dbaafc499b85)

python visualize_model_structure.py

![hybrid_cfd_model_structure](https://github.com/user-attachments/assets/0611eac8-98c7-48bb-89ab-6a3fc4f4bdba)

python visualize_output_slices.py

![output_pressure_heatmap](https://github.com/user-attachments/assets/931cde2f-f0d7-4327-a193-a8a7869a4a9e)
![output_velocity_quiver](https://github.com/user-attachments/assets/74a04ef2-ec57-4ba9-8f23-527b9ebcfbdf)

![output_pressure_surface3d](https://github.com/user-attachments/assets/968f057c-55e7-4a94-ab97-28ca0a008351)

# 开始训练模型
python model/train_fno3d.py
或者python train.py

📊 数据说明

DEM 地形数据：来源于 Copernicus（10m 分辨率）

风速数据：ERA5 再分析数据，2019~2024 年，U/V 风速 @ 10m、50m

植被信息：NDVI 或 GEDI 高度图，用于影响表面粗糙度建模

📈 模型结构

![hybrid_cfd_model pt](https://github.com/user-attachments/assets/eb8efde9-b06e-445b-a8bd-232070b97f50)


![model1](https://github.com/user-attachments/assets/496c18c0-2b37-42f0-a4e4-04b35452e00c)


基于 Fourier Neural Operator (FNO3D)和Tranformer

输入通道：DEM、NDVI、初始风速边界条件

输出通道：3D 风速张量场（u, v, w）

📬 联系方式

如需进一步交流，请联系维护者 Adolmeal 或提交 issue。
