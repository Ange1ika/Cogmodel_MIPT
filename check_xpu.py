import torch
import intel_extension_for_pytorch as ipex

print(torch.__version__)

if hasattr(torch, 'xpu') and torch.xpu.is_available():
    print(f'XPU devices count: {torch.xpu.device_count()}')
    for i in range(torch.xpu.get_device_count()):
        print(f'[{i}]: {torch.xpu.get_device_properties(i)}')
else:
    print("XPU devices are not available")