"""
d2l 2.2 — 数据预处理 (Data Preprocessing)

深度学习的数据预处理基本功：CSV 读取、缺失值处理、one-hot 编码、转为 tensor。
pandas 是 PyTorch 生态的标准伴侣——读表格数据几乎必用。
"""
import torch
import pandas as pd
import os

print("=" * 50)
print("2.2.1 — 读取 CSV 数据集")
print("=" * 50)

# 创建一个模拟数据集并写入文件
os.makedirs(os.path.dirname(__file__) + "/data", exist_ok=True)
data_file = os.path.join(os.path.dirname(__file__), "data", "house_tiny.csv")

with open(data_file, "w") as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,106000\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")

data = pd.read_csv(data_file)
print(data)
#    NumRooms Alley   Price
# 0       NaN  Pave  127500
# 1       2.0   NaN  106000
# 2       4.0   NaN  178100
# 3       NaN   NaN  140000


print("\n" + "=" * 50)
print("2.2.2 — 处理缺失值 (Missing Values)")
print("=" * 50)

# 策略 1: 插值法 —— 数值列用均值填充
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
print(f"填充前:\n{inputs}")

inputs = inputs.fillna(inputs.mean(numeric_only=True))
# 等价于 inputs.iloc[:, 0] = inputs.iloc[:, 0].fillna(inputs.iloc[:, 0].mean())
print(f"\n数值列 fillna(mean) 后:\n{inputs}")
# NumRooms 的 NaN → (2+4)/2 = 3.0

# 策略 2: 类别列用 one-hot 编码（见下一步）


print("\n" + "=" * 50)
print("2.2.3 — 类别变量转 One-Hot 编码")
print("=" * 50)

# pd.get_dummies: 自动把字符串列转成 one-hot 向量
inputs = pd.get_dummies(inputs, dummy_na=True, dtype=int)
print(f"get_dummies 后:\n{inputs}")
# Alley_Pave: 1 表示有巷道, 0 表示无
# Alley_nan:  1 表示该字段缺失（NaN）
# 三列: NumRooms, Alley_Pave, Alley_nan


print("\n" + "=" * 50)
print("2.2.4 — pandas DataFrame → PyTorch Tensor")
print("=" * 50)

# 最终把 pandas 数据转为 tensor（只能转数值）
X = torch.tensor(inputs.values, dtype=torch.float32)
y = torch.tensor(outputs.values, dtype=torch.float32)

print(f"X (features):\n{X}")
print(f"X.dtype = {X.dtype}, shape = {X.shape}")
print(f"y (labels) : {y}")


print("\n" + "=" * 50)
print("[OK] 2.2 数据预处理 — 完成")
print("=" * 50)
print("\n关键记忆点:")
print("  pd.read_csv()       → 读 CSV")
print("  df.fillna(mean())   → 数值缺失 → 均值填充")
print("  pd.get_dummies()    → 类别/NaN → one-hot 编码")
print("  torch.tensor(df.values) → DataFrame → Tensor")
