import torch
import torch.nn as nn
import torch.nn.functional as F


# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 残差块（包含两个卷积层 + 残差连接）
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=2 if downsample else 1,
            padding=1
        )
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.swish1 = Swish()
        
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.swish2 = Swish()
        
        self.residual = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=2 if downsample else 1,
                    padding=1
                ),
                nn.GroupNorm(num_groups=32, num_channels=out_channels)
            )
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.swish1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        out += residual
        out = self.swish2(out)
        
        return out


# VAE编码器 - 每个stage内部有跳跃连接
class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=128):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # 第一阶段
        self.stage1_res1 = ResNetBlock(base_channels, base_channels*2, downsample=False)
        self.stage1_res2 = ResNetBlock(base_channels*2, base_channels*2, downsample=False)
        self.stage1_down = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1)
        self.stage1_skip = nn.Conv2d(base_channels, base_channels*2, kernel_size=1)  # 跳跃连接调整层
        
        # 第二阶段
        self.stage2_res1 = ResNetBlock(base_channels*2, base_channels*4, downsample=False)
        self.stage2_res2 = ResNetBlock(base_channels*4, base_channels*4, downsample=False)
        self.stage2_down = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1)
        self.stage2_skip = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=1)
        
        # 第三阶段
        self.stage3_res1 = ResNetBlock(base_channels*4, base_channels*8, downsample=False)
        self.stage3_res2 = ResNetBlock(base_channels*8, base_channels*8, downsample=False)
        self.stage3_down = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, stride=2, padding=1)
        self.stage3_skip = nn.Conv2d(base_channels*4, base_channels*8, kernel_size=1)
        
        # 第四阶段
        self.stage4_res1 = ResNetBlock(base_channels*8, base_channels*8, downsample=False)
        self.stage4_res2 = ResNetBlock(base_channels*8, base_channels*8, downsample=False)
        self.stage4_skip = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=1)
        
        # 生成均值和方差
        self.latent_dim = 4
        self.mean = nn.Conv2d(base_channels*8, self.latent_dim, kernel_size=1)
        self.log_var = nn.Conv2d(base_channels*8, self.latent_dim, kernel_size=1)
    
    def forward(self, input):
        x = self.initial_conv(input)
        
        # 第一阶段
        stage1_res1_input = x
        x = self.stage1_res1(x)
        x = self.stage1_res2(x)
        x = x + self.stage1_skip(stage1_res1_input)  # 跳跃连接：第一个残差模块开始 → 第二个残差模块结束
        x = self.stage1_down(x)
        
        # 第二阶段
        stage2_res1_input = x
        x = self.stage2_res1(x)
        x = self.stage2_res2(x)
        x = x + self.stage2_skip(stage2_res1_input)  # 跳跃连接
        x = self.stage2_down(x)
        
        # 第三阶段
        stage3_res1_input = x
        x = self.stage3_res1(x)
        x = self.stage3_res2(x)
        x = x + self.stage3_skip(stage3_res1_input)  # 跳跃连接
        x = self.stage3_down(x)
        
        # 第四阶段
        stage4_res1_input = x
        x = self.stage4_res1(x)
        x = self.stage4_res2(x)
        x = x + self.stage4_skip(stage4_res1_input)  # 跳跃连接
        
        mean = self.mean(x)
        log_var = self.log_var(x)
        
        return mean, log_var


# VAE解码器 - 与编码器对称，每个stage内部也添加跳跃连接
class VAEDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=128, latent_dim=32):
        super().__init__()
        
        self.initial_conv = nn.Conv2d(latent_dim, base_channels*8, kernel_size=1)
        
        # 第一阶段（对应编码器第四阶段）
        self.stage1_res1 = ResNetBlock(base_channels*8, base_channels*8, downsample=False)
        self.stage1_res2 = ResNetBlock(base_channels*8, base_channels*8, downsample=False)
        self.stage1_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1)
        )
        self.stage1_skip = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=1)  # 跳跃连接调整层
        
        # 第二阶段（对应编码器第三阶段）
        self.stage2_res1 = ResNetBlock(base_channels*4, base_channels*4, downsample=False)
        self.stage2_res2 = ResNetBlock(base_channels*4, base_channels*4, downsample=False)
        self.stage2_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1)
        )
        self.stage2_skip = nn.Conv2d(base_channels*4, base_channels*4, kernel_size=1)
        
        # 第三阶段（对应编码器第二阶段）
        self.stage3_res1 = ResNetBlock(base_channels*2, base_channels*2, downsample=False)
        self.stage3_res2 = ResNetBlock(base_channels*2, base_channels*2, downsample=False)
        self.stage3_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1)
        )
        self.stage3_skip = nn.Conv2d(base_channels*2, base_channels*2, kernel_size=1)
        
        # 第四阶段（对应编码器第一阶段）
        self.stage4_res1 = ResNetBlock(base_channels, base_channels, downsample=False)
        self.stage4_res2 = ResNetBlock(base_channels, base_channels, downsample=False)
        self.stage4_skip = nn.Conv2d(base_channels, base_channels, kernel_size=1)
        
        # 最终卷积层
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, z):
        z = self.initial_conv(z)
        
        # 第一阶段
        stage1_res1_input = z
        z = self.stage1_res1(z)
        z = self.stage1_res2(z)
        z = z + self.stage1_skip(stage1_res1_input)  # 跳跃连接：第一个残差模块开始 → 第二个残差模块结束
        z = self.stage1_up(z)
        
        # 第二阶段
        stage2_res1_input = z
        z = self.stage2_res1(z)
        z = self.stage2_res2(z)
        z = z + self.stage2_skip(stage2_res1_input)  # 跳跃连接
        z = self.stage2_up(z)
        
        # 第三阶段
        stage3_res1_input = z
        z = self.stage3_res1(z)
        z = self.stage3_res2(z)
        z = z + self.stage3_skip(stage3_res1_input)  # 跳跃连接
        z = self.stage3_up(z)
        
        # 第四阶段
        stage4_res1_input = z
        z = self.stage4_res1(z)
        z = self.stage4_res2(z)
        z = z + self.stage4_skip(stage4_res1_input)  # 跳跃连接
        
        output = self.final_conv(z)
        return output


# 完整VAE模型
class VanillaVAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128):
        super().__init__()
        self.encode = VAEEncoder(in_channels, base_channels)
        self.decode = VAEDecoder(out_channels, base_channels, self.encode.latent_dim)
    
    def reparameterize(self, mean, log_var):
        """重参数化技巧：从正态分布中采样"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, input):
        # 编码
        mean, log_var = self.encode(input)
        
        # 重参数化
        z = self.reparameterize(mean, log_var)
        
        # 解码
        recon = self.decode(z)
        
        return recon, mean, log_var


# 测试模型
if __name__ == "__main__":
    # 创建模型实例
    model = VanillaVAE(in_channels=1, out_channels=1).cuda()
    
    # 测试输入
    input = torch.randn(1, 1, 256, 256).cuda()
    
    # 前向传播
    with torch.no_grad():
        recon, mean, log_var = model(input)
    
    # 打印输出形状
    print(f"输入形状: {input.shape}")
    print(f"重构形状: {recon.shape}")
    print(f"均值形状: {mean.shape}")
    print(f"对数方差形状: {log_var.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"总参数量: {total_params:.6f} Billion") # 0.159221 Billion
    