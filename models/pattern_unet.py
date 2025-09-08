import torch
import torch.nn as nn
import math

# 设置CrossAttention类
class CrossAttention(nn.Module):
    def __init__(self, input_channels: int, d_query: int, d_value: int, d_forward: int, d_condition: int):
        """
        初始化CrossAttention类
        :param input_channels:  输入特征通道数
        :param d_query:         查询向量维度
        :param d_value:         值向量维度
        :param d_forward:       前馈网络维度
        :param d_condition:     输入条件维度
        """
        # 继承父类初始化方法
        super(CrossAttention, self).__init__()
        
        # 设置线性投影层
        self.query_projection = nn.Linear(input_channels, d_query)
        self.key_projection = nn.Linear(d_condition, d_query)
        self.value_projection = nn.Linear(d_condition, d_value)
        self.output_projection = nn.Linear(d_value, input_channels)
        
        # 设置归一化层
        self.layer_norm1 = nn.LayerNorm(input_channels)
        self.layer_norm2 = nn.LayerNorm(input_channels)
        
        # 设置前馈层
        self.feedforward = nn.Sequential(
            nn.Linear(input_channels, d_forward),
            nn.ReLU(),
            nn.Linear(d_forward, input_channels)
        )
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        执行前向传播过程
        :param x:           输入特征
        :param condition:   输入条件
        :return:            输出特征
        """
        # 获取并调整输入特征的形状
        batch_size, input_channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1)
        
        # 执行线性投影层
        query = self.query_projection(x).view(batch_size, height * width, -1)
        key = self.key_projection(condition).unsqueeze(1)
        value = self.value_projection(condition).unsqueeze(1)
        
        # 执行注意力机制
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        attention = attention_scores.softmax(dim=-1)
        attention_output = torch.matmul(attention, value).view(batch_size, height, width, -1)
        context = self.output_projection(attention_output)
        
        # 执行前馈层
        x = self.layer_norm1(x + context)
        x = self.layer_norm2(x + self.feedforward(x))
        
        # 返回输出特征
        return x.permute(0, 3, 1, 2)
    
# 设置ConvolutionalBlock类
class ConvolutionalBlock(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, d_condition: int, use_attention: bool):
        """
        初始化ConvolutionalBlock类
        :param input_channels:      输入特征通道数
        :param output_channels:     输出特征通道数
        :param d_condition:         输入条件维度
        :param use_attention:       是否使用注意力机制
        """
        # 继承父类初始化方法
        super(ConvolutionalBlock, self).__init__()
        
        # 设置第一次卷积：仅改变特征通道数，不改变特征尺寸
        self.convolution1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.SiLU()
        )
        
        # 设置第二次卷积：不改变特征通道数也不改变特征尺寸
        self.convolution2 = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.SiLU()
        )
        
        # 设置残差连接
        self.residual = nn.Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0) if input_channels != output_channels else nn.Identity()

        # 设置交叉注意力机制
        self.use_attention = use_attention
        if use_attention:
            self.cross_attention = CrossAttention(output_channels, d_query=128, d_value=128, d_forward=output_channels * 4, d_condition=d_condition)
        else:
            self.condition_projection = nn.Linear(d_condition, output_channels)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor):
        """
        执行前向传播过程
        :param x:           输入特征
        :param condition:   输入条件
        :return:            输出特征
        """
        # 获取残差输入
        residual = self.residual(x)
        
        # 执行两次卷积
        x = self.convolution1(x)
        x = self.convolution2(x)
        
        # 执行残差连接
        x = x + residual

        # 执行交叉注意力机制
        if self.use_attention and condition is not None:
            x = self.cross_attention(x, condition)
        elif condition is not None:
            x = x + self.condition_projection(condition).unsqueeze(-1).unsqueeze(-1)
        
        # 返回输出特征
        return x
    
# 设置PatternUNet类
class PatternUNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, d_condition: int, use_attention: bool=True):
        """
        初始化PatternUNet类
        :param input_channels:      输入特征通道数
        :param output_channels:     输出特征通道数
        :param d_condition:         输入条件维度
        :param use_attention:       是否使用注意力机制
        """
        # 继承父类初始化方法
        super(PatternUNet, self).__init__()
        
        # 设置UNet的encoder结构
        self.encoder1 = ConvolutionalBlock(input_channels, 64, d_condition, use_attention)
        self.encoder2 = ConvolutionalBlock(64, 128, d_condition, use_attention)
        self.encoder3 = ConvolutionalBlock(128, 256, d_condition, use_attention)
        self.encoder4 = ConvolutionalBlock(256, 512, d_condition, use_attention)

        # 设置UNet的middle结构
        self.middle = ConvolutionalBlock(512, 1024, d_condition, use_attention)
        
        # 设置UNet的decoder结构
        self.decoder4 = ConvolutionalBlock(1024 + 512, 512, d_condition, use_attention)
        self.decoder3 = ConvolutionalBlock(512 + 256, 256, d_condition, use_attention)
        self.decoder2 = ConvolutionalBlock(256 + 128, 128, d_condition, use_attention)
        self.decoder1 = ConvolutionalBlock(128 + 64, 64, d_condition, use_attention)

        # 设置UNet的output结构
        self.output = nn.Conv2d(64, output_channels, kernel_size=1)
        
        # 设置池化层和上采样层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                    
    def forward(self, pattern_addnoised: torch.Tensor, transmittance: torch.Tensor, t: torch.Tensor):
        """
        执行前向传播过程
        :param pattern_addnoised:   加噪的超表面结构图像
        :param transmittance:        目标的超表面透射率
        :param t:                   当前时间步
        :return:                    预测的超表面结构图像噪声
        """
        # 获取输入参数的批大小
        batch_size = pattern_addnoised.shape[0]
        
        # 生成输入条件
        t_processed = torch.sin(t * math.pi / 2).view(batch_size, -1)
        condition = torch.cat([transmittance, t_processed], dim=1)
        
        # 执行UNet的encoder部分
        encoder_output1 = self.encoder1(pattern_addnoised, condition)
        encoder_output2 = self.encoder2(self.pool(encoder_output1), condition)
        encoder_output3 = self.encoder3(self.pool(encoder_output2), condition)
        encoder_output4 = self.encoder4(self.pool(encoder_output3), condition)
        
        # 执行UNet的middle部分
        middle_output = self.middle(self.pool(encoder_output4), condition)
        
        # 执行UNet的decoder部分
        decoder_output4 = self.decoder4(torch.cat([self.upsample(middle_output), encoder_output4], dim=1), condition)
        decoder_output3 = self.decoder3(torch.cat([self.upsample(decoder_output4), encoder_output3], dim=1), condition)
        decoder_output2 = self.decoder2(torch.cat([self.upsample(decoder_output3), encoder_output2], dim=1), condition)
        decoder_output1 = self.decoder1(torch.cat([self.upsample(decoder_output2), encoder_output1], dim=1), condition)
        
        # 执行UNet的output部分
        return self.output(decoder_output1)

# 设置测试样例
if __name__ == "__main__":
    # 设置测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for testing.")
    
    # 创建PatternUNet实例
    model = PatternUNet(input_channels=1, output_channels=1, d_condition=603, use_attention=True).to(device)
    
    # 设置输入参数
    batch_size = 32
    pattern_addnoised = torch.randn(batch_size, 1, 32, 32).to(device)
    transmittance = torch.randn(batch_size, 602).to(device)
    t = torch.randint(1, 1000, (batch_size, )).to(device)
    
    # 测试PatternUNet实例
    pattern_noise_predicted = model(pattern_addnoised, transmittance, t)
    print("PatternUNet输出的数据形状：", pattern_noise_predicted.shape)
    
    # 打印模型参数量
    total_params = sum(parameters.numel() for parameters in model.parameters() if parameters.requires_grad)
    print(f"模型总参数量: {total_params:,}")