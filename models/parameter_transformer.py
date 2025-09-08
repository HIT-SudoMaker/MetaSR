import torch
import torch.nn as nn

# 设置ParameterTransformer类
class ParameterTransformer(nn.Module):
    def __init__(self, n_heads: int, n_layers: int, dropout: float):
        """
        初始化ParameterTransformer类
        :param n_heads:     注意力头数量
        :param n_layers:    编码器层数量
        :param dropout:     随机失活概率
        """
        # 继承父类初始化方法
        super().__init__()
        
        # 设置Transformer输入层
        self.pattern_embedding = nn.Linear(1024, 1024)
        self.transmittance_embedding = nn.Linear(602, 1024)
        self.embedding_layer = nn.Linear(1024 + 1024, 1024)
        
        # 设置Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        
        # 设置Transformer前馈层
        self.feedforward = nn.Linear(1024, 3)
    
    def forward(self, pattern: torch.Tensor, transmittance: torch.Tensor):
        """
        执行前向传播过程
        :param pattern:         预测的超表面结构图像
        :param transmittance:   目标的超表面透射率
        :return:                预测的超表面结构参数
        """
        # 获取输入参数的批大小
        batch_size = pattern.shape[0]
        
        # 调整输入参数的形状
        pattern_embedded = self.pattern_embedding(pattern.contiguous().view(batch_size, -1))
        transmittance_embedded = self.transmittance_embedding(transmittance)
        src = torch.cat((pattern_embedded, transmittance_embedded), dim=1)
        
        # 执行Transformer输入层、编码器和前馈层
        src = self.embedding_layer(src)
        src = self.encoder(src)
        parameter_predicted = self.feedforward(src)
        
        # 返回预测的超表面结构参数
        return parameter_predicted

# 设置测试样例
if __name__ == "__main__":
    # 设置测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for testing.")
    
    # 创建ParameterTransformer实例
    model = ParameterTransformer(n_heads=16, n_layers=8, dropout=0.1).to(device)
    
    # 设置输入参数
    batch_size = 32
    pattern = torch.randn(batch_size, 1, 32, 32).to(device)
    transmittance = torch.randn(batch_size, 602).to(device)
    
    # 测试PrameterTransformer实例
    parameter_predicted = model(pattern, transmittance)
    print("PrameterTransformer输出的数据形状：", parameter_predicted.shape)
    
    # 打印模型参数量
    total_params = sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)
    print(f"模型总参数量: {total_params:,}")