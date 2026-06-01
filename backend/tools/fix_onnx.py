"""Auto-detect channel config from weights and build+export correct ONNX"""
import torch
import torch.nn as nn
import numpy as np
import cv2
import re


class ConvBNPReLU(nn.Module):
    def __init__(self, inp, oup, k=3, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x)))


class ConvBN(nn.Module):
    def __init__(self, inp, oup, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, oup, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(oup)

    def forward(self, x):
        return self.bn(self.conv(x))


class DepthWiseBlock(nn.Module):
    def __init__(self, inp, mid, oup):
        super().__init__()
        self.conv = ConvBNPReLU(inp, mid, k=1)
        self.conv_dw = ConvBNPReLU(mid, mid, k=3, s=2, p=1, g=mid)
        self.project = ConvBN(mid, oup, k=1)

    def forward(self, x):
        return self.project(self.conv_dw(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, inp, mid, oup):
        super().__init__()
        self.conv = ConvBNPReLU(inp, mid, k=1)
        self.conv_dw = ConvBNPReLU(mid, mid, k=3, s=1, p=1, g=mid)
        self.project = ConvBN(mid, oup, k=1)

    def forward(self, x):
        return x + self.project(self.conv_dw(self.conv(x)))


class ResidualBlocks(nn.Module):
    """Container using 'model' sub-module name to match official weight keys"""
    def __init__(self, blocks):
        super().__init__()
        self.model = nn.ModuleList(blocks)

    def forward(self, x):
        for blk in self.model:
            x = blk(x)
        return x


def parse_mid_channels(sd_keys, prefix):
    """Extract mid (expand) channels for each residual block from weight keys"""
    mids = []
    i = 0
    while True:
        key = f'module.{prefix}.model.{i}.conv.conv.weight'
        if key not in sd_keys:
            break
        mids.append(int(sd[key].shape[0]))  # out_channels of 1x1 conv = mid
        i += 1
    return mids


def mids_str(mids):
    return ', '.join(str(m) for m in mids)


# ============================================================
# Load weights and extract exact architecture
# ============================================================
print('Loading weights...')
sd = torch.load('models/2.7_80x80_MiniFASNetV2.pth', map_location='cpu',
                weights_only=True)
all_keys = set(sd.keys())

c1_out = int(sd['module.conv1.conv.weight'].shape[0])
c23_mid = int(sd['module.conv_23.conv.conv.weight'].shape[0])
c23_out = int(sd['module.conv_23.project.conv.weight'].shape[0])
m3 = parse_mid_channels(all_keys, 'conv_3')
c34_mid = int(sd['module.conv_34.conv.conv.weight'].shape[0])
c34_out = int(sd['module.conv_34.project.conv.weight'].shape[0])
m4 = parse_mid_channels(all_keys, 'conv_4')
c45_mid = int(sd['module.conv_45.conv.conv.weight'].shape[0])
c45_out = int(sd['module.conv_45.project.conv.weight'].shape[0])
m5 = parse_mid_channels(all_keys, 'conv_5')

print(f'Architecture:')
print(f'  conv1:     3 -> {c1_out}')
print(f'  conv_23:   {c1_out} -> {c23_mid} -> {c23_out}')
print(f'  conv_3:    {c23_out} x{len(m3)} blocks (mids=[{mids_str(m3)}]) -> {c23_out}')
print(f'  conv_34:   {c23_out} -> {c34_mid} -> {c34_out}')
print(f'  conv_4:    {c34_out} x{len(m4)} blocks (mids=[{mids_str(m4)}]) -> {c34_out}')
print(f'  conv_45:   {c34_out} -> {c45_mid} -> {c45_out}')
print(f'  conv_5:    {c45_out} x{len(m5)} blocks (mids=[{mids_str(m5)}]) -> {c45_out}')

# Check if prob has bias
has_prob_bias = 'module.prob.bias' in all_keys
print(f'  prob has bias: {has_prob_bias}')


# ============================================================
# Build model with extracted config
# ============================================================
class MiniFASNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        m = self.module = nn.Module()

        m.conv1 = ConvBNPReLU(3, c1_out, k=3, s=2, p=1)
        m.conv2_dw = ConvBNPReLU(c1_out, c1_out, k=3, s=1, p=1, g=c1_out)

        m.conv_23 = DepthWiseBlock(c1_out, c23_mid, c23_out)
        m.conv_3 = ResidualBlocks([ResidualBlock(c23_out, mid, c23_out) for mid in m3])
        m.conv_34 = DepthWiseBlock(c23_out, c34_mid, c34_out)
        m.conv_4 = ResidualBlocks([ResidualBlock(c34_out, mid, c34_out) for mid in m4])
        m.conv_45 = DepthWiseBlock(c34_out, c45_mid, c45_out)
        m.conv_5 = ResidualBlocks([ResidualBlock(c45_out, mid, c45_out) for mid in m5])

        m.conv_6_sep = ConvBNPReLU(c45_out, 512, k=1)
        m.conv_6_dw = ConvBN(512, 512, k=5, s=1, p=0, g=512)
        m.linear = nn.Linear(512, 128, bias=False)
        m.bn = nn.BatchNorm1d(128)
        m.prob = nn.Linear(128, num_classes, bias=has_prob_bias)

    def forward(self, x):
        m = self.module
        out = m.conv1(x)
        out = m.conv2_dw(out)
        out = m.conv_23(out)
        out = m.conv_3(out)
        out = m.conv_34(out)
        out = m.conv_4(out)
        out = m.conv_45(out)
        out = m.conv_5(out)
        out = m.conv_6_sep(out)
        out = m.conv_6_dw(out)
        out = out.flatten(1)
        out = m.linear(out)
        out = m.bn(out)
        out = m.prob(out)
        return out


print('\nBuilding model with extracted config...')
model = MiniFASNetV2(num_classes=3)
result = model.load_state_dict(sd, strict=True)

if result.missing_keys:
    print(f'MISSING ({len(result.missing_keys)}): {result.missing_keys[:3]}...')
else:
    print('All keys matched!')
if result.unexpected_keys:
    print(f'UNEXPECTED ({len(result.unexpected_keys)}): {result.unexpected_keys[:3]}...')


# ============================================================
# Test inference
# ============================================================
model.eval()

crop = cv2.imread('models/_debug_face_crop.jpg')
img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)
x = ((img_rgb - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
x_torch = torch.from_numpy(x)

with torch.no_grad():
    logits = model(x_torch)
    probs = torch.softmax(logits, dim=-1)
    am = probs[0].argmax().item()
    labels = ['0:live(real)', '1:spoof_print', '2:spoof_replay']
    print(f'\nReal face crop:')
    print(f'  logits={logits[0].tolist()}')
    print(f'  probs=[{probs[0][0]:.4f}, {probs[0][1]:.4f}, {probs[0][2]:.4f}]')
    print(f'  argmax={am} => {labels[am]}')

black = np.zeros_like(crop).astype(np.float32)
xb = ((black - 127.5) / 128.0).transpose(2, 0, 1)[np.newaxis]
with torch.no_grad():
    ob = model(torch.from_numpy(xb))
    pb = torch.softmax(ob, dim=-1)
    pbm = pb[0].argmax().item()
    print(f'\nBlack image baseline:')
    print(f'  probs=[{pb[0][0]:.4f}, {pb[0][1]:.4f}, {pb[0][2]:.4f}]')
    print(f'  argmax={pbm} => {labels[pbm]}')

if am != pbm or abs(probs[0][am] - pb[0][pbm]) > 0.1:
    print('\n*** Model can distinguish real face from black! ***')


# ============================================================
# Export correct ONNX
# ============================================================
print('\nExporting ONNX...')
torch.onnx.export(
    model, x_torch,
    'models/minifasnet_v2_correct.onnx',
    input_names=['input'], output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}},
    opset_version=18,
)
import os
sz = os.path.getsize('models/minifasnet_v2_correct.onnx')
print(f'Saved: models/minifasnet_v2_correct.onnx ({sz/1024:.0f} KB)')

import onnxruntime as ort
sess = ort.InferenceSession(
    'models/minifasnet_v2_correct.onnx',
    providers=['CPUExecutionProvider']
)
r_onnx = sess.run(None, {'input': x_torch.numpy()})[0]
diff = abs(r_onnx[0] - logits[0].numpy()).max()
print(f'ONNX verify: max_diff={diff:.8f}')
if diff < 1e-4:
    print('SUCCESS! Correct ONNX model exported and verified.')
