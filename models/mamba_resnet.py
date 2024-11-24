import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

def get_inplanes():
    # return [32, 64, 128, 256]  # 原来是[64, 128, 256, 512]
    return [64, 128, 256, 512] 

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
class SelectiveScan(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.d_state = max(d_state, 16)
        self.d_model = d_model
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_state * 2)  # 2倍宽度用于门控
        self.out_proj = nn.Linear(self.d_state, d_model)
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(self.d_state, self.d_state))
        self.B = nn.Parameter(torch.randn(self.d_state, self.d_state))
        self.C = nn.Parameter(torch.randn(self.d_state, self.d_state))
        self.D = nn.Parameter(torch.randn(self.d_state))
        
        # Mamba style normalization
        self.norm = nn.LayerNorm(d_model)  # 输入归一化
        self.x_proj_weight = nn.Parameter(torch.randn(self.d_state, d_model))
        self.dt_proj_weight = nn.Parameter(torch.randn(self.d_state, d_model))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Mamba style initialization
        nn.init.normal_(self.A, mean=0, std=0.01)
        self.A.data = -torch.exp(self.A)  
        nn.init.normal_(self.B, mean=0, std=0.01)
        nn.init.normal_(self.C, mean=0, std=0.01)
        nn.init.normal_(self.D, mean=0, std=0.01)
        
        # Initialize projection weights
        nn.init.normal_(self.x_proj_weight, mean=0, std=0.01)
        nn.init.normal_(self.dt_proj_weight, mean=0, std=0.01)
        
    def forward(self, x):
        b, c, t, h, w = x.shape
        
        # Apply input normalization first (Mamba style)
        x = x.permute(0, 3, 4, 1, 2).contiguous()  # (b, h, w, c, t)
        x = x.view(b * h * w, c, t)  # (b*h*w, c, t)
        
        # Input normalization
        x_norm = self.norm(x.transpose(-1, -2)).transpose(-1, -2)
        
        # Initialize state
        state = torch.zeros(b * h * w, self.d_state).to(x.device)
        outputs = []
        
        for ti in range(t):
            # Get current input
            xt = x_norm[:, :, ti]  # (b*h*w, c)
            
            # Project and split into x_proj and dt (Mamba style)
            projected = self.in_proj(xt)  # (b*h*w, 2*d_state)
            x_proj, gate = projected.chunk(2, dim=-1)  # Each (b*h*w, d_state)
            
            # Apply gating (Mamba style)
            gate = torch.sigmoid(gate)
            
            # State update with dt-scaling (Mamba style)
            state_update = F.linear(x_proj, self.A)
            state = state * torch.sigmoid(self.D)[None, :] + state_update * gate
            
            # Output computation
            state_out = F.linear(state, self.B)
            xt_out = F.linear(x_proj, self.C)
            combined = state_out + xt_out
            
            # Output projection
            out = self.out_proj(combined)
            outputs.append(out)
        
        # Reshape output
        out = torch.stack(outputs, dim=2)  # (b*h*w, c, t)
        out = out.view(b, h, w, c, t)  # (b, h, w, c, t)
        out = out.permute(0, 3, 4, 1, 2)  # (b, c, t, h, w)
        
        return out

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveScan(d_model, d_state)
        self.proj = nn.Linear(d_model, d_model)
        
        # 初始化线性层
        nn.init.trunc_normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)
        
    def forward(self, x):
        # x shape: (batch, channels, time, height, width)
        b, c, t, h, w = x.shape
        
        # 重塑以应用LayerNorm
        residual = x
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # (b, t, h, w, c)
        x = x.view(-1, c)  # (b*t*h*w, c)
        
        # 应用LayerNorm
        x = self.norm(x)
        x = x.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)  # (b, c, t, h, w)
        
        # SSM处理
        ssm_out = self.ssm(x)
        
        # 线性投影
        proj_in = x.permute(0, 2, 3, 4, 1).contiguous()  # (b, t, h, w, c)
        proj_in = proj_in.view(-1, c)
        proj_out = self.proj(proj_in)
        proj_out = proj_out.view(b, t, h, w, c).permute(0, 4, 1, 2, 3)
        
        # 组合输出
        out = ssm_out + proj_out + residual
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.mamba = MambaBlock(d_model=planes, d_state=planes//4)  # 添加Mamba块
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
                 conv1_t_stride=2,
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
if __name__ == "__main__":
    model = generate_model(model_depth=34,
                                      n_classes=2,
                                      n_input_channels=1)
    x = torch.randn(8, 1, 30, 224, 224)  # (batch, channels, time, height, width)
    output = model(x)
    print(output.shape)  # torch.Size([2, 400])