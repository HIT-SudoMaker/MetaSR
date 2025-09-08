import torch
import torch.nn as nn

# 设置infer_parameter过程
def infer_parameter(pattern_inferred: torch.Tensor, transmittance_target: torch.Tensor, model: nn.Module, device: torch.device):
    """
    推理超表面结构参数
    :param pattern_inferred:        推理的超表面结构图案
    :param transmittance_targeted:   目标的超表面透射率
    :param model:                   模型
    :param device:                  计算设备
    :return:                        推理的超表面结构参数
    """
    # 调整超表面结构图案和透射率的形状和设备
    pattern_inferred = pattern_inferred.to(device)
    transmittance_target = transmittance_target.unsqueeze(0).repeat(pattern_inferred.shape[0], 1).to(device)
    
    # 推理超表面结构参数
    with torch.no_grad():
        parameter_inferred = model(pattern_inferred, transmittance_target)
    
    # 返回推理的超表面结构参数
    return parameter_inferred