import torch
from torch import tensor


causal_mask = tensor([
         [ True, False, False, False, False],
         [ True,  True, False, False, False],
         [ True,  True,  True, False, False],
         [ True,  True,  True,  True, False],
         [ True,  True,  True,  True,  True]
        ])

attention_mask = tensor([
         [0., 0., 1., 1., 1.],
         [1., 1., 1., 1., 1.]
        ])

print(causal_mask.unsqueeze(0).unsqueeze(0))
print(attention_mask.unsqueeze(1).unsqueeze(1).to(torch.bool))

print(
    (
                causal_mask.unsqueeze(0).unsqueeze(0) &
                attention_mask.unsqueeze(1).unsqueeze(1).to(torch.bool)
            )
)