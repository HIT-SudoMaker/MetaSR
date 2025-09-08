import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import pandas as pd
from dataset.dataset import MetasurfaceDataset
from diffusion.diffusion import Diffusion
from models.pattern_unet import PatternUNet

# 设置train_pattern_unet过程
def train_pattern_unet(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, n_epochs: int, learning_rate: float, n_timesteps: int, model: nn.Module):
    """
    训练PatternUnet模型
    :param train_dataset:   训练集
    :param val_dataset:     验证集
    :param batch_size:      批次大小
    :param n_epochs:        训练轮数
    :param learning_rate:   初始学习率
    :param n_timesteps:     时间步数
    :param model:           神经网络模型
    :return:                训练完毕的模型和损失
    """
    # 设置训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training.")
    
    # 加载模型至训练设备
    diffusion = Diffusion(n_timesteps, device)
    model.to(device)
    
    # 设置数据集加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, persistent_workers=True, pin_memory=True)
    
    # 设置训练优化器
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=2e-6)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.85)
    criterion = nn.MSELoss()
    
    # 设置训练损失和验证损失
    train_losses = []
    val_losses = []
    
    # 设置最佳验证损失和保存神经网络模型的起始轮数
    best_val_loss = float('inf')
    start_epoch = 100
    best_epoch = 0
    best_model_state_dict = None
    
    # 设置单轮次训练过程
    for epoch in range(n_epochs):
        # 设置模型为训练模式
        model.train()
        train_loss_total = 0
        
        # 设置单迭代训练过程
        for batch in tqdm(train_loader):
            # 设置优化器梯度清零
            optimizer.zero_grad()
            
            # 加载数据集至训练设备
            real = batch["real"].to(device)
            imag = batch["imag"].to(device)
            pattern = batch["pattern"].to(device)
            
            # 执行前向加噪过程
            t = torch.randint(1, n_timesteps, (pattern.shape[0], ), dtype=torch.int32).to(device)
            pattern_addnoised, pattern_noise = diffusion.addnoise(pattern, None, t)
            
            # 生成透射率提示词
            transmittance = torch.cat((real, imag), dim=1).to(device)
            
            # 执行前向传播过程
            pattern_noise_predicted = model(pattern_addnoised, transmittance, t)
            
            # 计算训练损失
            loss = criterion(pattern_noise_predicted, pattern_noise)
            
            # 执行反向传播过程
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            # 添加训练损失
            train_loss_total += loss.item()
        
        # 记录单轮次训练过程
        scheduler.step()
        train_loss_average = train_loss_total / len(train_loader)
        train_losses.append(train_loss_average)
        
        # 设置模型为验证模式
        model.eval()
        val_loss_total = 0
        
        # 设置单迭代验证过程
        with torch.no_grad():
            for batch in tqdm(val_loader):
                # 加载数据集至训练设备
                real = batch["real"].to(device)
                imag = batch["imag"].to(device)
                pattern = batch["pattern"].to(device)
                
                # 执行前向加噪过程
                t = torch.randint(1, n_timesteps, (pattern.shape[0], ), dtype=torch.int32).to(device)
                pattern_addnoised, pattern_noise = diffusion.addnoise(pattern, None, t)
                
                # 生成透射率提示词
                transmittance = torch.cat((real, imag), dim=1).to(device)
                
                # 执行前向传播过程
                pattern_noise_predicted = model(pattern_addnoised, transmittance, t)
                
                # 计算验证损失
                loss = criterion(pattern_noise_predicted, pattern_noise)
                
                # 添加验证损失
                val_loss_total += loss.item()
        
        # 记录单轮次验证过程
        val_loss_average = val_loss_total / len(val_loader)
        val_losses.append(val_loss_average)
        
        # 打印单轮次训练过程
        print(f"Epoch [{epoch + 1}/{n_epochs}], Train Loss: {train_loss_average:.8f}, Val Loss: {val_loss_average:.8f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # 在起始轮数后保存最佳神经网络模型
        if epoch >= start_epoch - 1 and val_loss_average < best_val_loss:  # 修正条件
            best_val_loss = val_loss_average
            best_epoch = epoch + 1
            best_model_state_dict = copy.deepcopy(model.state_dict())
            print(f"Best model saved at epoch {best_epoch}, Val Loss: {best_val_loss:.8f}.")
        
    # 加载最佳神经网络模型
    if best_model_state_dict is not None:
        model.load_state_dict(best_model_state_dict)
        print(f"Training Completed. Best model saved at epoch {best_epoch}, Val Loss: {best_val_loss:.8f}")
    else:
        print(f"Training Completed. No best model saved, Final Val Loss: {val_losses[-1]:.8f}.")
    
    # 返回训练完毕的模型和损失
    return model, train_losses, val_losses
            
# 主程序
if __name__ == "__main__":
    # 设置超参数
    dataset_path = "dataset/freeform.mat"
    batch_size = 128
    n_epochs = 200
    learning_rate = 5e-5
    n_timesteps = 1000
    model_name = "model05"
    
    # 创建保存目录
    os.makedirs("results/pattern_unet/models", exist_ok=True)
    os.makedirs("results/pattern_unet/losses", exist_ok=True)
    
    # 设置结果保存路径
    model_save_path = os.path.join("results/pattern_unet/models", f"{model_name}.pth")
    loss_save_path = os.path.join("results/pattern_unet/losses", f"{model_name}.csv")
    
    # 加载并划分数据集
    dataset = MetasurfaceDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(3407))
    
    # 创建模型
    model = PatternUNet(input_channels=1, output_channels=1, d_condition=603, use_attention=True)
    
    # 训练模型
    best_model, train_losses, val_losses = train_pattern_unet(train_dataset, val_dataset, batch_size, n_epochs, learning_rate, n_timesteps, model)
    
    # 保存模型
    torch.save(best_model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}.")
    
    # 保存损失
    losses = {
        "train_losses": train_losses,
        "val_losses": val_losses
    }
    df = pd.DataFrame(losses)
    df.to_csv(loss_save_path, index=False)
    print(f"Losses saved to {loss_save_path}.")
    
    # 输出训练完成信息
    print("\nTraining Record:")
    print("="*60)
    print(f"Final Train Loss: {train_losses[-1]:.8f}")
    print(f"Final Val Loss: {val_losses[-1]:.8f}")
    print(f"Best Val Loss: {min(val_losses):.8f} (Epoch {val_losses.index(min(val_losses)) + 1})")
    print("="*60)
    print("Training Completed.")