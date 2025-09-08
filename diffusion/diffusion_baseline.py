import torch
import torch.nn as nn

# 设置Diffusion类
class Diffusion:
    def __init__(self, n_timesteps: int, device: torch.device):
        """
        初始化Diffusion类
        :param n_timesteps: 时间步数
        :param device:      计算设备
        """
        # 获取扩散过程的控制参数
        self.n_timesteps = n_timesteps
        self.device = device
        
        # 生成beta参数序列和alpha参数序列
        self.beta = torch.linspace(1e-4, 2e-2, self.n_timesteps, device=self.device)
        self.alpha = 1 - self.beta
        
        # 生成扩散过程的额外参数序列
        self.sqrt_recip_alpha = torch.sqrt(1 / self.alpha)
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
    
    def addnoise(self, pattern: torch.Tensor, pattern_noise: torch.Tensor, t: torch.Tensor):
        """
        执行前向加噪过程：x_0 -> x_t
        :param pattern:         初始的超表面结构图案
        :param pattern_noise:   可选的超表面结构噪声
        :param t:               当前时间步
        :return:                加噪后的超表面结构图案和使用的超表面结构噪声
        """
        # 生成超表面结构噪声
        if pattern_noise is None:
            pattern_noise = torch.randn_like(pattern, device=self.device)
        
        # 获取当前时间步的参数值，并调整形状
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        # 计算并返回加噪后的超表面结构图案和使用的超表面结构噪声
        return sqrt_alpha_bar_t * pattern + sqrt_one_minus_alpha_bar_t * pattern_noise, pattern_noise
    
    @torch.no_grad()
    def denoise(self, pattern_addnoised: torch.Tensor, parameter: torch.Tensor, transmittance: torch.Tensor, t: torch.Tensor, model: nn.Module):
        """
        执行单步反向去噪过程：x_t -> x_t-1
        :param pattern_addnoised:   加噪后的超表面结构图案
        :param parameter:           超表面结构参数
        :param transmittance:       超表面透射率
        :param t:                   当前时间步
        :param model:               噪声预测模型
        :return:                    单步反向去噪后的超表面结构图案
        """
        # 获取当前时间步的参数值，并调整形状
        beta_t = self.beta[t].view(-1, 1, 1, 1)
        sqrt_recip_alpha_t = self.sqrt_recip_alpha[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        
        # 计算单步反向去噪后的超表面结构图案
        pattern_noise_predicted = model(pattern_addnoised, parameter, transmittance, t)
        pattern_denoised = sqrt_recip_alpha_t * (pattern_addnoised - beta_t * pattern_noise_predicted / sqrt_one_minus_alpha_bar_t)
        
        # 生成随机噪声
        pattern_noise = torch.randn_like(pattern_denoised, device=self.device)
        
        # 返回单步反向去噪后的超表面结构图案
        return torch.where(t.view(-1, 1, 1, 1) > 0, pattern_denoised + torch.sqrt(beta_t) * pattern_noise, pattern_denoised)
    
    @torch.no_grad()
    def infer(self, pattern_initial: torch.Tensor, parameter: torch.Tensor, transmittance: torch.Tensor, model: nn.Module, threshold: float=0.51):
        """
        执行完整反向去噪过程：x_T -> x_0
        :param pattern_initial:     初始化的超表面结构图案
        :param parameter:           超表面结构参数
        :param transmittance:       超表面透射率
        :param model:               噪声预测模型
        :param threshold:           超表面结构图案二值化阈值
        :return:                    完整反向去噪后的超表面结构图案
        """
        # 获取初始化的超表面结构图案的批次大小
        batch_size = pattern_initial.shape[0]
        
        # 计算完整反向去噪后的超表面结构图案
        pattern_denoised = pattern_initial
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((batch_size,), i, device=self.device)
            pattern_denoised = self.denoise(pattern_denoised, parameter, transmittance, t, model)
        pattern_denoised = (torch.tanh(pattern_denoised) + 1) / 2
        pattern_denoised = (pattern_denoised > threshold).float()
        
        # 计算完整反向去噪后的超表面结构图案
        return pattern_denoised

# 设置测试样例
if __name__ == "__main__":
    # 设置测试设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for testing.")
    
    # 创建Diffusion实例
    n_timesteps = 1000
    diffusion = Diffusion(n_timesteps, device)
    
    # 设置输入参数
    pattern = torch.randn(32, 1, 32, 32).to(diffusion.device)
    parameter = torch.randn(32, 3).to(device)
    transmittance = torch.randn(32, 602).to(diffusion.device)
    t = torch.randint(1, n_timesteps, (32, )).to(diffusion.device)
    
    # 设置SimpleModel类
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, pattern_addnoised, parameter, transmittance, t):
            return torch.randn_like(pattern_addnoised)
    model = SimpleModel().to(diffusion.device)
    
    # 测试addnoise方法
    pattern_addnoised, pattern_noise = diffusion.addnoise(pattern, None, t)
    print("addnoise方法输出的数据形状：")
    print("pattern_addnoised:", pattern_addnoised.shape)
    print("pattern_noise:", pattern_noise.shape)
    print()
    
    # 测试denoise方法
    pattern_denoised = diffusion.denoise(pattern_addnoised, parameter, transmittance, t, model)
    print("denoise方法输出的数据形状：")
    print("pattern_denoised:", pattern_denoised.shape)
    print()
    
    # 测试infer方法
    pattern_inferred = diffusion.infer(pattern_addnoised, parameter, transmittance, model)
    print("infer方法输出的数据形状：")
    print("pattern_inferred:", pattern_inferred.shape)