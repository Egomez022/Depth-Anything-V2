import torch
import torchprofile
import os
import cv2
import numpy as np
import sys
import warnings
import time

from torchinfo import summary
from fvcore.nn import FlopCountAnalysis
from depth_anything_v2.dpt import DepthAnythingV2

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Compute Parameters and FLOPS for Depth-Anything-V2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

encoder = 'vitl'
model2 = DepthAnythingV2(**model_configs[encoder])
model2.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'), strict=False)
model2 = model2.to(DEVICE).eval()

# Compute FLOPS
inputTensor = torch.randn(1, 3, 392, 392).to(DEVICE)
depth_anything_flops = torchprofile.profile_macs(model2, args=(inputTensor,))

# Compute Parameters for Depth-Anything-V2
depth_anything_summary = summary(model2, input_size=(1, 3, 392, 392), verbose=0)
print(f"Depth-Anything-V2 - Parameters: {depth_anything_summary.total_params:,}")
print(f"Depth-Anything-V2 - FLOPS: {depth_anything_flops / 1e9:.2f} GFLOPS")

# Measure total operations (FLOPS)
operations = torchprofile.profile_macs(model2, args=(inputTensor,))
print(f"Total Operations (FLOPS + approximate ops): {operations / 1e9:.2f} GigaOps")

# Measure inference time
start_time = time.time()
with torch.no_grad():
    model2(inputTensor)  # Run a single inference
end_time = time.time()

inference_time = end_time - start_time
print(f"Inference Time: {inference_time:.4f} seconds")

# Calculate TOPS
tops = operations / (inference_time * 1e12)
print(f"Estimated TOPS: {tops:.2f} TOPS")