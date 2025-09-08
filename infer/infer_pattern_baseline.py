import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from diffusion.diffusion_baseline import Diffusion

# 设置infer_pattern过程
def infer_pattern(parameter_target: torch.Tensor, transmittance_target: torch.Tensor, n_samples: int, n_timesteps: int, model: nn.Module, device: torch.device, crop_indicator: bool=True, drc: bool=True):
    """
    推理超表面结构图案
    :param parameter:               目标的超表面结构参数
    :param transmittance_target:    目标的超表面透射率
    :param n_samples:               目标的样本生成数量
    :param n_timesteps:             时间步数
    :param model:                   模型
    :param device:                  计算设备
    :param crop_indicator:          裁剪指示符
    :param drc:                     是否执行设计规则检查
    :return:                        推理的超表面结构图案
    """
    # 调整超表面透射率的形状和设备
    parameter_target = parameter_target.unsqueeze(0).repeat(n_samples, 1).to(device)
    transmittance_target = transmittance_target.unsqueeze(0).repeat(n_samples, 1).to(device)
    
    # 生成随机噪声
    if crop_indicator:
        initial_noise = torch.randn((n_samples, 1, 32, 32), device=device)
    else:
        initial_noise = torch.randn((n_samples, 1, 64, 64), device=device)
    
    # 初始化Diffusion模型
    diffusion = Diffusion(n_timesteps, device)
    
    # 推理超表面结构图案
    pattern_inferred = diffusion.infer(initial_noise, parameter_target, transmittance_target, model)
    
    # 设置设计规则检查
    if drc:
        # 创建列表存储合法结果和索引
        valid_patterns = []
        
        # 执行设计规则检查
        for i in range(n_samples):
            # 将张量转换为numpy数组进行连通域分析
            pattern = pattern_inferred[i, 0].detach().cpu().numpy()
            labeled, n_features = ndimage.label(pattern)
        
            # 检查连通域数量和尺寸是否满足要求
            max_domains, min_domain_size = 8, 10
            if n_features > max_domains or (n_features > 0 and any(size < min_domain_size for size in ndimage.sum(pattern, labeled, range(1, n_features + 1)))):
                continue
        
            # 添加合法结果
            valid_patterns.append(pattern_inferred[i])
        
        # 如果没有合法结果，打印警告并返回原始结果
        if not valid_patterns:
            print("Warning: No valid patterns found after DRC!")
            return pattern_inferred
        
        # 如果有合法结果，返回合法结果
        return torch.stack(valid_patterns).to(device)
    
    else:
        return pattern_inferred