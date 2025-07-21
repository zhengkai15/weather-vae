import torch
import torch.nn as nn
import torch.nn.functional as F


# Swish激活函数 (f(x) = x * sigmoid(x))
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# 残差块（包含两个卷积层 + 残差连接）
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        
        # 第一个卷积层（可能进行下采样）
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=2 if downsample else 1,  # 仅在需要下采样时stride=2
            padding=1
        )
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.swish1 = Swish()
        
        # 第二个卷积层（保持尺寸不变）
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            stride=1, 
            padding=1
        )
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels)
        self.swish2 = Swish()
        
        # 残差连接（处理通道数和尺寸的变化）
        self.residual = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=3, 
                    stride=2 if downsample else 1,  # 与主路径保持一致
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
        
        out += residual  # 残差连接
        out = self.swish2(out)
        
        return out


# VAE编码器
class VAEEncoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=128):
        super().__init__()
        
        # 初始1×1×C卷积层
        self.initial_conv = nn.Conv2d(in_channels, base_channels, kernel_size=1)
        
        # 第一阶段（两个ResNet块 + 下采样）
        self.stage1 = nn.Sequential(
            ResNetBlock(base_channels, base_channels*2, downsample=False),
            ResNetBlock(base_channels*2, base_channels*2, downsample=False),
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, stride=2, padding=1)  # 下采样块
        )
        
        # 第二阶段（两个ResNet块 + 下采样）
        self.stage2 = nn.Sequential(
            ResNetBlock(base_channels*2, base_channels*4, downsample=False),
            ResNetBlock(base_channels*4, base_channels*4, downsample=False),
            nn.Conv2d(base_channels*4, base_channels*4, kernel_size=3, stride=2, padding=1)  # 下采样块
        )
        
        # 第三阶段（两个ResNet块 + 下采样）
        self.stage3 = nn.Sequential(
            ResNetBlock(base_channels*4, base_channels*8, downsample=False),
            ResNetBlock(base_channels*8, base_channels*8, downsample=False),
            nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, stride=2, padding=1)  # 下采样块
        )
        
        # 第四阶段（两个ResNet块，无下采样）
        self.stage4 = nn.Sequential(
            ResNetBlock(base_channels*8, base_channels*8, downsample=False),
            ResNetBlock(base_channels*8, base_channels*8, downsample=False)
        )
        
        # 生成均值和方差
        self.mean = nn.Conv2d(base_channels*8, 4, kernel_size=1)  # 输出通道数为4
        self.log_var = nn.Conv2d(base_channels*8, 4, kernel_size=1)  # 输出通道数为4
    
    def forward(self, input):
        x = self.initial_conv(input)
        
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        
        mean = self.mean(x)
        log_var = self.log_var(x)
        
        return mean, log_var


# VAE解码器（与编码器对称）
class VAEDecoder(nn.Module):
    def __init__(self, out_channels=3, base_channels=128):
        super().__init__()
        
        # 初始将通道维度从4转换为512
        self.initial_conv = nn.Conv2d(4, base_channels*8, kernel_size=1)
        
        # 第一阶段（两个ResNet块 + 上采样）
        self.stage1 = nn.Sequential(
            ResNetBlock(base_channels*8, base_channels*8, downsample=False),
            ResNetBlock(base_channels*8, base_channels*8, downsample=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 双线性插值上采样
            nn.Conv2d(base_channels*8, base_channels*4, kernel_size=3, padding=1)  # 调整通道数
        )
        
        # 第二阶段（两个ResNet块 + 上采样）
        self.stage2 = nn.Sequential(
            ResNetBlock(base_channels*4, base_channels*4, downsample=False),
            ResNetBlock(base_channels*4, base_channels*4, downsample=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 双线性插值上采样
            nn.Conv2d(base_channels*4, base_channels*2, kernel_size=3, padding=1)  # 调整通道数
        )
        
        # 第三阶段（两个ResNet块 + 上采样）
        self.stage3 = nn.Sequential(
            ResNetBlock(base_channels*2, base_channels*2, downsample=False),
            ResNetBlock(base_channels*2, base_channels*2, downsample=False),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  # 双线性插值上采样
            nn.Conv2d(base_channels*2, base_channels, kernel_size=3, padding=1)  # 调整通道数
        )
        
        # 第四阶段（两个ResNet块，无上采样）
        self.stage4 = nn.Sequential(
            ResNetBlock(base_channels, base_channels, downsample=False),
            ResNetBlock(base_channels, base_channels, downsample=False)
        )
        
        # 最终卷积层，将通道维度转换为输出维度
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
    
    def forward(self, z):
        z = self.initial_conv(z)
        
        z = self.stage1(z)
        z = self.stage2(z)
        z = self.stage3(z)
        z = self.stage4(z)
        
        output = self.final_conv(z)
        return output


# 完整VAE模型
class VanillaVAE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=128):
        super().__init__()
        self.encode = VAEEncoder(in_channels, base_channels)
        self.decode = VAEDecoder(out_channels, base_channels)
    
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
    
    # # 打印模型结构
    # print(model)
    
    # 测试输入（假设输入尺寸为548×860）
    input = torch.randn(1, 1, 512, 512).cuda()
    
    # 前向传播
    with torch.no_grad():
        recon, mean, log_var = model(input)
    
    # 打印输出形状
    print(f"输入形状: {input.shape}")
    print(f"重构形状: {recon.shape}")
    print(f"均值形状: {mean.shape}") # torch.Size([1, 4, 64, 64]) # 16倍压缩
    print(f"对数方差形状: {log_var.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Total parameters: {total_params} Billion") # 0.156087049 Billion