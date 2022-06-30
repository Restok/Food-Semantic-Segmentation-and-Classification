# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

x1 = torch.ones(1,11,267,242)
x2 = torch.ones(1,10)
x2 = nn.ConstantPad1d((1,0), 0)(x2)
x2 = x2.repeat_interleave(x1.size()[-1]*x1.size()[-2]).view(x1.size())
x1 = x1+x2
print(x1[0, :, -1,-1])