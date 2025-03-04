import torch
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

model = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2').cuda()
dummy_input = torch.randn(1, 3, 512, 512).cuda()

with torch.no_grad():
    model(dummy_input)
torch.cuda.synchronize()
