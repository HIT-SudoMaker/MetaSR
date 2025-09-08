import torch
import torch.nn as nn

# 设置TransmittanceTransformer类
class TransmittanceTransformer(nn.Module):
    def __init__(self, n_heads: int, n_layers: int, dropout: float):
        """
        初始化TransmittanceTransformer类
        :param n_heads:     注意力头数量
        :param n_layers:    编码器层数量
        :param dropout:     随机失活概率
        """
        # 继承父类初始化方法
        super().__init__()
        
        # 设置Transformer输入层
        self.pattern_embedding = nn.Linear(1024, 1024)
        self.parameter_embedding = nn.Linear(3, 1024)
        self.embedding_layer = nn.Linear(1024 + 1024, 4096)
        
        # 设置Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=4096, nhead=n_heads, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=n_layers)
        
        # 设置Transformer前馈层
        self.feedforward_real = nn.Linear(4096, 301)
        self.feedforward_imag = nn.Linear(4096, 301)
    
    def forward(self, pattern: torch.Tensor, parameter: torch.Tensor):
        """
        执行前向传播过程
        :param pattern:     预测的超表面结构图像
        :param parameter:   预测的超表面结构参数
        :return:            预测的超表面透射率实部和虚部
        """
        # 获取输入参数的批大小
        batch_size = pattern.shape[0]
        
        # 调整输入参数的形状
        pattern_embedded = self.pattern_embedding(pattern.contiguous().view(batch_size, -1))
        parameter_embedded = self.parameter_embedding(parameter)
        src = torch.cat((pattern_embedded, parameter_embedded), dim=1)
        
        # 执行Transformer输入层、编码器和前馈层
        src = self.embedding_layer(src)
        src = self.encoder(src)
        real_predicted = self.feedforward_real(src)
        imag_predicted = self.feedforward_imag(src)
        
        # 返回预测的超表面透射率实部和虚部
        return real_predicted, imag_predicted

# 设置测试样例
if __name__ == "__main__":
    # 设置测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for testing.")
    
    # 创建TransmittanceTransformer实例
    model = TransmittanceTransformer(n_heads=16, n_layers=8, dropout=0.1).to(device)
    
    # 设置输入参数
    batch_size = 32
    pattern = torch.randn(batch_size, 1, 32, 32).to(device)
    parameter = torch.randn(batch_size, 3).to(device)
    
    # 创建TransmittanceTransformer实例
    real_predicted, imag_predicted = model(pattern, parameter)
    print("TransmittanceTransformer输出的数据形状：", real_predicted.shape)
    print("TransmittanceTransformer输出的数据形状：", imag_predicted.shape)
    
    # 打印模型参数量
    total_params = sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)
    print(f"模型总参数量: {total_params:,}")