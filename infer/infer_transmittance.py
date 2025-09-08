import torch
import torch.nn as nn

# 设置infer_transmittance过程
def infer_transmittance(pattern_inferred: torch.Tensor, parameter_inferred: torch.Tensor, model: nn.Module, device: torch.device):
    """
    推理超表面透射率
    :param pattern_inferred:        推理的超表面结构图案
    :param parameter_inferred:      推理的超表面透射率
    :param model:                   模型
    :param device:                  计算设备
    :return:                        推理的超表面透射率
    """
    # 调整超表面结构图案和结构参数的形状和设备
    pattern_inferred = pattern_inferred.to(device)
    parameter_inferred = parameter_inferred.to(device)
    
    # 推理超表面结构参数
    with torch.no_grad():
        tranmission_real_inferred, tranmission_imag_predicted = model(pattern_inferred, parameter_inferred)
    
    # 返回推理的超表面透射率
    return tranmission_real_inferred, tranmission_imag_predicted