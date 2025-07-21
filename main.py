# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from era5 import ERA5
import yaml
from models.vanilla_vae import VanillaVAE
import matplotlib.pyplot as plt
import numpy as np

# %%
yaml_file_path = "./conf/config.yaml"
with open(yaml_file_path, "r") as f:
    config = yaml.safe_load(f)
train_dataset = ERA5(config=config, flag="train"
)
batchsize=4
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batchsize,
    shuffle=True,
    num_workers=4
)

test_dataset = ERA5(config=config, flag="test"
)
test_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1
)

# %%
train_dataset.__getitem__(1).shape

# %%

vae = VanillaVAE(in_channels=1, out_channels=1)
# vae = torch.load('vae_model.pt')
if torch.cuda.is_available():
    vae.cuda()
    
input = torch.randn((1, 1, 512, 512)).cuda()
output = vae(input=input)
print(output[1].shape)
print(output[0].shape)

output = vae.encode(input=input)
print(output[0].shape)

# %%
lr=1e-5
optimizer = optim.Adam(vae.parameters(), lr=lr)
# CosineAnnealingLR调度器
num_epochs = 51
total_batches = len(train_loader) * num_epochs  # 总batch数
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=total_batches,  # 学习率下降的总batch数
    eta_min=1e-6          # 学习率的最小值
)


# return reconstruction error + KL divergence losses
beta = 0.01
def loss_function(recon_x, x, mu, log_var):
    recons_loss =F.mse_loss(recon_x, x)
    KLD = torch.mean(torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)) * beta
    return recons_loss.item(), KLD.item(), recons_loss + KLD


# %%
def train(epoch):
    vae.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.cuda()
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = vae(data)
        recons_loss, KLD, loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        # 每个batch后更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
    
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{:06d}/{:06d}]\tLR: {:.6f}\tLoss: {:.6f}\trecons_loss: {:.6f}\tKLD: {:.6f}'.format(
                epoch, batchsize * batch_idx, len(train_loader.dataset),current_lr, loss.item(), recons_loss, KLD))
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# %%
def test():
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.cuda()
            recon, mu, log_var = vae(data)
            
            # sum up batch loss
            test_loss += loss_function(recon, data, mu, log_var).item()
        
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# %%
# 创建包含两个子图的画布
def denormlize(data):
    data = data * 3.890266 + 296.5198
    return data

# %%
for epoch in range(1, num_epochs):
    train(epoch)
    # test()
    
    vae.eval()
    test_loss= 0
    with torch.no_grad():
        for data in test_loader:
            data = data.cuda()
            sample, mu, log_var = vae(data)
            tensor_tmp_output = sample.view(1, 1, 512, 512).cpu()
            data = data.cpu()
            
            data_origin = denormlize(data)
            data_pred = denormlize(tensor_tmp_output) 

            # 创建包含三个子图的画布
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            for i in range(data_origin.shape[0]):
                data_origin_cur = data_origin[i, 0]
                data_pred_cur = data_pred[i, 0]
                
                # 清空并重置子图
                for ax in axes:
                    ax.clear()
                
                # 设置等高线级别
                levels = np.arange(250, 320, 2)
                
                # 计算差异场和RMSE
                diff = data_pred_cur - data_origin_cur
                rmse = float(torch.sqrt(torch.mean(diff ** 2)))
                
                # 绘制第一个子图：原始数据
                im1 = axes[0].contourf(np.flipud(data_origin_cur), levels=levels)
                axes[0].set_aspect('equal')
                axes[0].set_title('origin')
                
                # 绘制第二个子图：预测数据
                im2 = axes[1].contourf(np.flipud(data_pred_cur), levels=levels)
                axes[1].set_aspect('equal')
                axes[1].set_title('pred')
                
                # 绘制第三个子图：差异场（使用蓝红颜色映射突出正负差异）
                diff_levels = np.arange(-15, 16, 1)  # 差异场范围可根据实际情况调整
                im3 = axes[2].contourf(np.flipud(diff), levels=diff_levels, cmap='bwr')
                axes[2].set_aspect('equal')
                axes[2].set_title(f' (pred - origin RMSE: {rmse:.2f})')
                
                # 调整布局
                plt.tight_layout()
                
                # 添加共享的颜色条
                cbar1 = fig.colorbar(im1, ax=axes[:2].ravel().tolist(), shrink=0.95)
                
                # 差异场单独的颜色条
                cbar2 = fig.colorbar(im3, ax=axes[2], shrink=0.95)
                cbar2.set_label('diff')

                # 显示图片（可选，若在服务器环境可注释掉）
                # plt.show()
                plt.savefig(f"./sample_v7.finetune.beta_0.01.lr_1e-5_cosine/epoch_{epoch:03d}.png", dpi = 100)
                
                # 关闭当前画布
                plt.close()    
                break
            break

    # %%
    # 模型保存
    torch.save(vae, f'vae_model.finetune.beta_0.01.lr_1e-5_cosine_{lr}.pt')  # 保存整个模型

# %%
# ## 加载模型
# # vae = torch.load('vae_model_v1.pt')

vae.eval()
test_loss= 0
with torch.no_grad():
    for data in test_loader:
        data = data.cuda()
        sample, mu, log_var = vae(data)
        tensor_tmp_output = sample.view(1, 1, 512, 512).cpu()
        data = data.cpu()
        
        data_origin = denormlize(data)
        data_pred = denormlize(tensor_tmp_output) 

        # 创建包含三个子图的画布
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for i in range(data_origin.shape[0]):
            data_origin_cur = data_origin[i, 0]
            data_pred_cur = data_pred[i, 0]
            
            # 清空并重置子图
            for ax in axes:
                ax.clear()
            
            # 设置等高线级别
            levels = np.arange(250, 320, 2)
            
            # 计算差异场和RMSE
            diff = data_pred_cur - data_origin_cur
            rmse = float(torch.sqrt(torch.mean(diff ** 2)))
            
            # 绘制第一个子图：原始数据
            im1 = axes[0].contourf(np.flipud(data_origin_cur), levels=levels)
            axes[0].set_aspect('equal')
            axes[0].set_title('origin')
            
            # 绘制第二个子图：预测数据
            im2 = axes[1].contourf(np.flipud(data_pred_cur), levels=levels)
            axes[1].set_aspect('equal')
            axes[1].set_title('pred')
            
            # 绘制第三个子图：差异场（使用蓝红颜色映射突出正负差异）
            diff_levels = np.arange(-15, 16, 1)  # 差异场范围可根据实际情况调整
            im3 = axes[2].contourf(np.flipud(diff), levels=diff_levels, cmap='bwr')
            axes[2].set_aspect('equal')
            axes[2].set_title(f' (pred - origin RMSE: {rmse:.2f})')
            
            # 调整布局
            plt.tight_layout()
            
            # 添加共享的颜色条
            cbar1 = fig.colorbar(im1, ax=axes[:2].ravel().tolist(), shrink=0.95)
            
            # 差异场单独的颜色条
            cbar2 = fig.colorbar(im3, ax=axes[2], shrink=0.95)
            cbar2.set_label('diff')

            # 显示图片（可选，若在服务器环境可注释掉）
            # plt.show()
            
            # 关闭当前画布
            plt.close()    
            break
        break


