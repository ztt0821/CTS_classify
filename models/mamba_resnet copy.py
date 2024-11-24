import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

def get_inplanes():
    return [32, 64, 128, 256]  # 原来是[64, 128, 256, 512]
    # return [64, 128, 256, 512] 

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     dilation=dilation,
                     stride=stride,
                     padding=dilation,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

# class MambaBlock(nn.Module):
#     def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_inner = int(d_model * expand_factor)
#         self.d_hidden = self.d_inner // 2

#         # 添加层归一化
#         self.norm = nn.LayerNorm(d_model)
        
        
#         # Projects
#         self.in_proj = nn.Linear(d_model, self.d_inner)
#         nn.init.xavier_uniform_(self.in_proj.weight, gain=0.02)
#         nn.init.zeros_(self.in_proj.bias)
        
#         self.conv1d = nn.Conv1d(
#             in_channels=self.d_hidden,
#             out_channels=self.d_hidden,
#             kernel_size=d_conv,
#             groups=self.d_hidden,
#             padding='same'
#         )
#         nn.init.xavier_uniform_(self.conv1d.weight, gain=0.02)
#         nn.init.zeros_(self.conv1d.bias)
        
#         # SSM parameters
#         self.A = nn.Parameter(torch.randn(self.d_hidden, d_state) / d_state**0.5)
#         self.B = nn.Parameter(torch.randn(self.d_hidden, d_state))
#         self.C = nn.Parameter(torch.randn(self.d_hidden, d_state))
#         self.D = nn.Parameter(torch.randn(self.d_state))
#         # print(f"A shape: {self.A.shape}")
#         # print(f"B shape: {self.B.shape}")
#         # print(f"C shape: {self.C.shape}")
#         # print(f"D shape: {self.D.shape}")
        
#         self.dt_proj = nn.Linear(self.d_hidden, self.d_hidden)

        
#         # self.activation = nn.SiLU()
#         self.activation = nn.GELU()
#         self.out_proj = nn.Linear(self.d_inner, d_model)
#         nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
#         nn.init.zeros_(self.out_proj.bias)

#     def check_nan(self, tensor, name):
#         if torch.isnan(tensor).any():
#             print(f"NaN detected in {name}")
#             print(f"Shape: {tensor.shape}")
#             print(f"Min: {tensor[~torch.isnan(tensor)].min() if torch.sum(~torch.isnan(tensor)) > 0 else 'all NaN'}")
#             print(f"Max: {tensor[~torch.isnan(tensor)].max() if torch.sum(~torch.isnan(tensor)) > 0 else 'all NaN'}")
#             print(f"Mean: {tensor[~torch.isnan(tensor)].mean() if torch.sum(~torch.isnan(tensor)) > 0 else 'all NaN'}")
#             print(f"NaN count: {torch.isnan(tensor).sum().item()}")
#             print(f"Total elements: {tensor.numel()}")
#             return True
#         return False
#     def forward(self, x):
#         device = x.device
#         dtype = x.dtype
#                 # 添加数值检查
#         # 添加层归一化
#         # x = self.norm(x)
#         # 重排维度以使用 LayerNorm
#         identity = x
#         b, c, d, h, w = x.shape
#         x = x.reshape(b, c, -1)  # [B, C, D*H*W]
#         x = x.transpose(1, 2)    # [B, D*H*W, C]
#         x = self.norm(x)         # 在 channel 维度上进行归一化
#         x = x.transpose(1, 2)    # [B, C, D*H*W]
#         x = x.reshape(b, c, d, h, w)  # 恢复原始形状
#         L = d * h * w
#         x = x.reshape(b, c, -1).transpose(1, 2)  # [B, L, C]
        
#         x_and_res = self.in_proj(x)  # [B, L, d_inner]
#         x, res = x_and_res.chunk(2, dim=-1)

        
#         conv_x = x.transpose(-1, -2)

#         conv_x = self.conv1d(conv_x)
#                 # 5. 检查卷积后


#         x = conv_x.transpose(-1, -2)

#         x = self.activation(x)
 
        
#         delta = self.dt_proj(x)
#         delta = self.activation(delta)


#         # 保存原始batch维度的x用于后续concat
#         x_orig = x
        
#         # 准备SSM参数
#         # 维度转换：把batch维度放到最后
#         x = x.transpose(0, 1)  # [L, B, d_hidden]
#         delta = delta.transpose(0, 1)  # [L, B, d_hidden]
        
#         A = self.A.transpose(0, 1).contiguous()  # 从 [d_hidden, d_state] 变为 [d_state, d_hidden]
#         B = self.B.transpose(0, 1).contiguous()
#         C = self.C.transpose(0, 1).contiguous()
#         D = self.D.contiguous()
        
#         # 添加更详细的维度信息打印
#         print("\nDetailed shape information:")
#         print(f"x: shape={x.shape}, dim order=['L', 'B', 'd_hidden']")
#         print(f"delta: shape={delta.shape}, dim order=['L', 'B', 'd_hidden']")
#         print(f"A: shape={A.shape}, dim order=['d_state', 'd_hidden']")
#         print("delta", delta.shape)
#         print("B", B.shape)
#         print("C", C.shape)
#         print("D", D.shape)
#         # print(f"Expected A shape: ({x.shape[-1]}, {self.d_state})")
        
#         x_ssm = selective_scan_fn(
#             x,          # [L, B, d_hidden]
#             delta,      # [L, B, d_hidden]
#             A,         # [d_state, d_hidden]
#             B,         # [d_state, d_hidden]
#             C,         # [d_state, d_hidden]
#             D,    # [d_state]
#             None,      # z
#             None,      # delta_bias
#             True,      # delta_softplus
#             False      # return_last_state
#         )
#                     # 8. 检查SSM输出
        
#         # 转换回原来的维度顺序
#         x_ssm = x_ssm.transpose(0, 1)  # [B, L, d_hidden]
#         x = x.transpose(0, 1)  # [B, L, d_hidden]
 
#         x = torch.cat([x_ssm, x_orig], dim=-1)

#         x = self.out_proj(x)

#         x = x + res
        
#         x = x.transpose(1, 2).reshape(b, c, d, h, w)
#         return x
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand_factor=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand_factor)
        self.d_hidden = self.d_inner // 2

        # 添加层归一化
        self.norm = nn.LayerNorm(d_model)
        
        
        # Projects
        self.in_proj = nn.Linear(d_model, self.d_inner)
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.02)
        nn.init.zeros_(self.in_proj.bias)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_hidden,
            out_channels=self.d_hidden,
            kernel_size=d_conv,
            groups=self.d_hidden,
            padding='same'
        )
        nn.init.xavier_uniform_(self.conv1d.weight, gain=0.02)
        nn.init.zeros_(self.conv1d.bias)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_hidden, d_state) / d_state**0.5)
        self.B = nn.Parameter(torch.randn(self.d_hidden, d_state))
        self.C = nn.Parameter(torch.randn(self.d_hidden, d_state))
        self.D = nn.Parameter(torch.randn(self.d_hidden))
        # print(f"A shape: {self.A.shape}")
        # print(f"B shape: {self.B.shape}")
        # print(f"C shape: {self.C.shape}")
        # print(f"D shape: {self.D.shape}")
        
        self.dt_proj = nn.Linear(self.d_hidden, self.d_hidden)

        
        # self.activation = nn.SiLU()
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(self.d_inner, d_model)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def check_nan(self, tensor, name):
        if torch.isnan(tensor).any():
            print(f"NaN detected in {name}")
            print(f"Shape: {tensor.shape}")
            print(f"Min: {tensor[~torch.isnan(tensor)].min() if torch.sum(~torch.isnan(tensor)) > 0 else 'all NaN'}")
            print(f"Max: {tensor[~torch.isnan(tensor)].max() if torch.sum(~torch.isnan(tensor)) > 0 else 'all NaN'}")
            print(f"Mean: {tensor[~torch.isnan(tensor)].mean() if torch.sum(~torch.isnan(tensor)) > 0 else 'all NaN'}")
            print(f"NaN count: {torch.isnan(tensor).sum().item()}")
            print(f"Total elements: {tensor.numel()}")
            return True
        return False
    def forward(self, x):
        device = x.device
        dtype = x.dtype
                # 添加数值检查
        # 添加层归一化
        # x = self.norm(x)
        # 重排维度以使用 LayerNorm
        identity = x
        b, c, d, h, w = x.shape
        x = x.reshape(b, c, -1)  # [B, C, D*H*W]
        x = x.transpose(1, 2)    # [B, D*H*W, C]
        x = self.norm(x)         # 在 channel 维度上进行归一化
        x = x.transpose(1, 2)    # [B, C, D*H*W]
        x = x.reshape(b, c, d, h, w)  # 恢复原始形状
        L = d * h * w
        x = x.reshape(b, c, -1).transpose(1, 2)  # [B, L, C]
        
        x_and_res = self.in_proj(x)  # [B, L, d_inner]
        x, res = x_and_res.chunk(2, dim=-1)

        
        conv_x = x.transpose(-1, -2)

        conv_x = self.conv1d(conv_x)
                # 5. 检查卷积后


        x = conv_x.transpose(-1, -2)

        x = self.activation(x)
 
        
        delta = self.dt_proj(x)
        delta = self.activation(delta)


        # 保存原始batch维度的x用于后续concat
        x_orig = x
        
        # 准备SSM参数
        # 维度转换：把batch维度放到最后
        x = x.transpose(0, 1)  # [L, B, d_hidden]
        delta = delta.transpose(0, 1)  # [L, B, d_hidden]
        
        A = self.A.contiguous()  # 从 [d_hidden, d_state] 变为 [d_state, d_hidden]
        B = self.B.contiguous()
        C = self.C.contiguous()
        D = self.D.contiguous()
        
        # 添加更详细的维度信息打印
        print("\nDetailed shape information:")
        print(f"x: shape={x.shape}, dim order=['L', 'B', 'd_hidden']")
        print(f"delta: shape={delta.shape}, dim order=['L', 'B', 'd_hidden']")
        print(f"A: shape={A.shape}, dim order=['d_state', 'd_hidden']")
        print("delta", delta.shape)
        print("B", B.shape)
        print("C", C.shape)
        print("D", D.shape)
        # print(f"Expected A shape: ({x.shape[-1]}, {self.d_state})")
        
        x_ssm = selective_scan_fn(
            x,          # [L, B, d_hidden]
            delta,      # [L, B, d_hidden]
            A,         # [d_state, d_hidden]
            B,         # [d_state, d_hidden]
            C,         # [d_state, d_hidden]
            D,    # [d_state]
            None,      # z
            None,      # delta_bias
            True,      # delta_softplus
            False      # return_last_state
        )
                    # 8. 检查SSM输出
        
        # 转换回原来的维度顺序
        x_ssm = x_ssm.transpose(0, 1)  # [B, L, d_hidden]
        x = x.transpose(0, 1)  # [B, L, d_hidden]
 
        x = torch.cat([x_ssm, x_orig], dim=-1)

        x = self.out_proj(x)

        x = x + res
        
        x = x.transpose(1, 2).reshape(b, c, d, h, w)
        return x
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.mamba = MambaBlock(planes)  # 添加Mamba块
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        out = self.mamba(out)  # 应用Mamba处理

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.mamba = MambaBlock(planes * self.expansion)  # 添加Mamba块
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.mamba(out)  # 应用Mamba处理

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class MambaResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 dilation=[1,1,1,1],
                 stride=[2,2,2],
                 n_classes=400):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type, dilation=dilation[0])
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=stride[0], 
                                       dilation=dilation[1])
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=stride[1], 
                                       dilation=dilation[2])
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=stride[2], 
                                       dilation=dilation[3])

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.faltten = nn.Flatten()
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  dilation=dilation,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.faltten(x)

        # x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = MambaResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = MambaResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = MambaResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = MambaResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = MambaResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = MambaResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = MambaResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model