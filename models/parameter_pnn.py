import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 设置TransmittanceProcessingNetwork类（适配32×32输入）
class TransmittanceProcessingNetwork(nn.Module):
    def __init__(self, input_dim: int = 602, hidden_dim: int = 256, output_dim: int = 64):
        """
        初始化TransmittanceProcessingNetwork类
        :param input_dim:   输入传输系数维度（2×301展平后为602）
        :param hidden_dim:  隐藏层维度
        :param output_dim:  输出特征维度
        """
        # 继承父类初始化方法
        super(TransmittanceProcessingNetwork, self).__init__()
        
        # 设置多层感知机处理传输系数
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True)
        )
        
        # 空间扩展层（调整为4×4以匹配图像处理网络的输出）
        self.spatial_expansion = nn.ConvTranspose2d(
            output_dim, output_dim, 
            kernel_size=4, stride=1, padding=0
        )
        
    def forward(self, transmittance: torch.Tensor):
        """
        执行前向传播过程
        :param transmittance:   输入传输系数 [batch_size, 602]
        :return:               输出特征 [batch_size, 64, 4, 4]
        """
        batch_size = transmittance.shape[0]
        
        # 通过MLP处理
        features = self.mlp(transmittance)  # [batch_size, 64]
        
        # 重塑为1×1特征图
        features = features.view(batch_size, -1, 1, 1)  # [batch_size, 64, 1, 1]
        
        # 空间扩展到4×4
        features = self.spatial_expansion(features)  # [batch_size, 64, 4, 4]
        
        return features

# 设置ImageProcessingNetwork类（适配32×32输入）
class ImageProcessingNetwork(nn.Module):
    def __init__(self, input_channels: int = 1):
        """
        初始化ImageProcessingNetwork类
        :param input_channels:  输入图像通道数
        """
        # 继承父类初始化方法
        super(ImageProcessingNetwork, self).__init__()
        
        # 设置第一个卷积块：32×32 -> 16×16
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 设置第二个卷积块：16×16 -> 8×8
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 设置第三个卷积块：8×8 -> 4×4
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor):
        """
        执行前向传播过程
        :param x:   输入图像 [batch_size, 1, 32, 32]
        :return:    输出特征 [batch_size, 64, 4, 4]
        """
        # CNN1, 2 + Pooling 1, 2
        x = self.conv_block1(x)  # [batch_size, 32, 32, 32]
        x = self.pool1(x)        # [batch_size, 32, 16, 16]
        x = self.conv_block2(x)  # [batch_size, 32, 16, 16]
        x = self.pool2(x)        # [batch_size, 32, 8, 8]
        
        # CNN3 + Pooling 3
        x = self.conv_block3(x)  # [batch_size, 64, 8, 8]
        x = self.pool3(x)        # [batch_size, 64, 4, 4]
        
        return x

# 修改后的ParameterPredictionNetwork类（适配4×4输入）
class ParameterPredictionNetwork(nn.Module):
    def __init__(self, input_channels: int = 128, output_dim: int = 3):
        """
        初始化ParameterPredictionNetwork类
        :param input_channels:  输入特征通道数（融合后的特征）
        :param output_dim:      输出维度（材料索引、厚度、晶格尺寸）
        """
        # 继承父类初始化方法
        super(ParameterPredictionNetwork, self).__init__()
        
        # 由于输入现在是4×4，我们简化网络结构
        # 设置CNN4-5层（保持4×4×128）
        self.conv_block4_5 = nn.Sequential(
            # CNN4
            nn.Conv2d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # CNN5
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # CNN6: 4×4×128 -> 4×4×256
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Pooling: 4×4 -> 2×2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Sequential(
            nn.Linear(2 * 2 * 256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # 输出层（预测3个参数）
        self.fc3 = nn.Linear(128, output_dim)
        
        # 输出激活
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        """
        执行前向传播过程
        :param x:   输入特征 [batch_size, 128, 4, 4]
        :return:    输出预测 [batch_size, 3] (材料索引、厚度、晶格尺寸)
        """
        # CNN4-5层
        x = self.conv_block4_5(x)  # [batch_size, 128, 4, 4]
        
        # CNN6
        x = self.conv_block6(x)    # [batch_size, 256, 4, 4]
        
        # Pooling
        x = self.pool(x)           # [batch_size, 256, 2, 2]
        
        # 展平
        x = x.view(x.size(0), -1)  # [batch_size, 1024]
        
        # 全连接层
        x = self.fc1(x)            # [batch_size, 256]
        x = self.fc2(x)            # [batch_size, 128]
        x = self.fc3(x)            # [batch_size, 3]
        
        # 输出激活
        output = self.output_activation(x)
        
        return output

# 设置ParameterPNN类
class ParameterPNN(nn.Module):
    def __init__(self, transmittance_dim: int = 301, pattern_size: int = 32, output_dim: int = 3):
        """
        初始化ParameterPNN类
        :param transmittance_dim:   传输系数维度（每个部分）
        :param pattern_size:        输入图案尺寸
        :param output_dim:          输出维度（3个参数）
        """
        # 继承父类初始化方法
        super(ParameterPNN, self).__init__()
        
        # 设置传输系数处理网络
        self.transmittance_network = TransmittanceProcessingNetwork(
            input_dim=2 * transmittance_dim,  # 实部+虚部
            hidden_dim=256,
            output_dim=64
        )
        
        # 设置2D图像处理网络
        self.image_network = ImageProcessingNetwork(input_channels=1)
        
        # 设置参数预测网络
        self.parameter_prediction = ParameterPredictionNetwork(
            input_channels=128,  # 64 + 64
            output_dim=output_dim
        )
        
    def forward(self, pattern: torch.Tensor, transmittance: torch.Tensor):
        """
        执行前向传播过程
        :param pattern:         Meta-atom图像 [batch_size, 1, 32, 32]
        :param transmittance:   传输系数 [batch_size, 602] (实部和虚部已展平)
        :return:               预测的参数 [batch_size, 3]
        """
        # 传输系数处理
        transmittance_features = self.transmittance_network(transmittance)  # [batch_size, 64, 4, 4]
        
        # 2D图像处理
        image_features = self.image_network(pattern)  # [batch_size, 64, 4, 4]
        
        # 特征融合（堆叠）
        combined_features = torch.cat([image_features, transmittance_features], dim=1)  # [batch_size, 128, 4, 4]
        
        # 参数预测
        predicted_parameters = self.parameter_prediction(combined_features)  # [batch_size, 3]
        
        return predicted_parameters

# 设置测试样例
if __name__ == "__main__":
    # 设置测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for testing.")
    
    # 创建ParameterPNN实例
    model = ParameterPNN(transmittance_dim=301, pattern_size=32, output_dim=3).to(device)
    
    # 设置输入参数
    batch_size = 32
    pattern = torch.randn(batch_size, 1, 32, 32).to(device)
    transmittance = torch.randn(batch_size, 602).to(device)
    
    # 测试ParameterPNN实例
    predicted_parameters = model(pattern, transmittance)
    print("ParameterPNN输出的数据形状：", predicted_parameters.shape)
    
    # 打印模型参数量
    total_params = sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)
    print(f"模型总参数量: {total_params:,}")