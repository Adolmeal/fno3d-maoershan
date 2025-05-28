fno3d-maoershan: 林区风场 FNO 替代模型

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
python visualize_model_structure.py
python visualize_output_slices.py

# 开始训练模型
python model/train_fno3d.py
或者python train.py

📊 数据说明

DEM 地形数据：来源于 Copernicus（10m 分辨率）

风速数据：ERA5 再分析数据，2019~2024 年，U/V 风速 @ 10m、50m

植被信息：NDVI 或 GEDI 高度图，用于影响表面粗糙度建模

📈 模型结构
![hybrid_cfd_model_structure](https://github.com/user-attachments/assets/47b556db-691d-4da2-88df-4a7fa41fb53b)


基于 Fourier Neural Operator (FNO3D)

输入通道：DEM、NDVI、初始风速边界条件

输出通道：3D 风速张量场（u, v, w）

📬 联系方式

如需进一步交流，请联系维护者 Adolmeal 或提交 issue。
