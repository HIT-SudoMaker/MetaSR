import torch
import torch.nn as nn
from infer.infer_pattern import infer_pattern
from infer.infer_parameter import infer_parameter
from infer.infer_transmittance import infer_transmittance

# 设置infer_all过程
def infer_all(transmittance_target: torch.Tensor, n_samples: int, n_timesteps: int, pattern_model: nn.Module, parameter_model: nn.Module, transmittance_model: nn.Module, device: torch.device, crop_indicator: bool=True, drc: bool=True):
    """
    依次推理超表面结构图案、结构参数和透射率
    :param transmittance_target:    目标的超表面透射率
    :param n_samples:               目标的样本生成数量
    :param n_timesteps:             时间步数
    :param pattern_model:           超表面结构图案推理模型
    :param parameter_model:         超表面结构参数推理模型
    :param transmittance_model:     超表面透射率推理模型
    :param device:                  推理设备
    :param crop_indicator:          裁剪指示符
    :param drc:                     是否执行设计规则检查
    :return:                        推理得到的超表面结构图案、结构参数和透射率
    """
    # 依次推理超表面结构图案、结构参数和透射率
    with torch.no_grad():
        pattern_inferred = infer_pattern(transmittance_target, n_samples, n_timesteps, pattern_model, device, crop_indicator, drc)
        parameter_inferred = infer_parameter(pattern_inferred, transmittance_target, parameter_model, device)
        transmittance_real_inferred, transmittance_imag_inferred = infer_transmittance(pattern_inferred, parameter_inferred, transmittance_model, device)
    
    # 返回推理得到的超表面结构图案、结构参数和透射率
    return pattern_inferred, parameter_inferred, transmittance_real_inferred, transmittance_imag_inferred
