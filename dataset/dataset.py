import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

# 设置MetasurfaceDataset类
class MetasurfaceDataset(Dataset):
    def __init__(self, matfile_path: str, crop_indicator: bool=True):
        """
        初始化MetasurfaceDataset类
        :param matfile_path:    数据集路径
        :param crop_indicator:  裁剪指示符
        """
        # 加载数据集
        try:
            data = loadmat(matfile_path)
            self.real = data["real"]
            self.imag = data["imag"]
            self.parameter = data["parameter"]
            self.pattern = data["pattern"]
        except KeyError as e:
            raise KeyError(f"Key error occurred while loading data: {e}") from e
        except Exception as e:
            raise Exception(f"An error occurred while loading data: {e}") from e
        
        # 处理parameter数据
        # parameter第一列为介于2.5和3之间的Lattice size
        # parameter第二列为介于0.5和1之间的Thickness
        # parameter第三列为介于3.5和5之间的Refractive index
        self.parameter[:, 0] = (self.parameter[:, 0] - 2.5) / 0.5
        self.parameter[:, 1] = (self.parameter[:, 1] - 0.5) / 0.5
        self.parameter[:, 2] = (self.parameter[:, 2] - 3.5) / 1.5
        
        # 处理pattern数据
        self.pattern = np.transpose(self.pattern, (2, 0, 1))
        if crop_indicator:
            self.pattern = self.pattern[:, :32, :32]
        
        # 将所有数据转换为张量
        self.real = torch.from_numpy(self.real.astype(np.float32))
        self.imag = torch.from_numpy(self.imag.astype(np.float32))
        self.parameter = torch.from_numpy(self.parameter.astype(np.float32))
        self.pattern = torch.from_numpy(self.pattern.astype(np.float32)).unsqueeze(1)
    
    def __getitem__(self, idx: int):
        """
        获取数据集在索引位置的样本
        :param idx: 索引位置
        """
        return {
            "real": self.real[idx],
            "imag": self.imag[idx],
            "parameter": self.parameter[idx],
            "pattern": self.pattern[idx]
        }
        
    def __len__(self):
        """
        获取数据集样本数量
        """
        return len(self.real)
    
# 设置测试样例
if __name__ == "__main__":
    # 创建MetasurfaceDataset实例
    matfile_path = "dataset/freeform.mat"
    dataset = MetasurfaceDataset(matfile_path, True)
    
    # 测试MetasurfaceDataset实例
    print(f"dataset length: {len(dataset)}")
    sample_data = dataset[1]
    print("Sample data tensor structure:")
    print(f"real: {sample_data["real"].shape}")
    print(f"imag: {sample_data["imag"].shape}")
    print(f"parameter: {sample_data["parameter"].shape}")
    print(f"pattern: {sample_data["pattern"].shape}")
    print(f"min: {torch.min(sample_data["pattern"])}, max: {torch.max(sample_data["pattern"])}, mean: {torch.mean(sample_data["pattern"])}")