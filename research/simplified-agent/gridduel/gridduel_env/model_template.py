"""Model template for TorchModelPolicy

Expected input: float tensor of shape [B, C=4, H, W] with values in {0,1}.
Output: logits over 9 actions (shape [B, 9]) or a scalar discrete action.

Example usage to save a checkpoint compatible with TorchModelPolicy:

    import torch
    model = TinyNet()
    torch.save(model, "models/agent.pt")

or using TorchScript:

    scripted = torch.jit.script(model)
    scripted.save("models/agent.pt")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyNet(nn.Module):
	def __init__(self, in_channels: int = 4, num_actions: int = 9):
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
		self.head = nn.Sequential(
			nn.Flatten(),
			nn.Linear(32 * 7 * 7, 128),
			nn.ReLU(),
			nn.Linear(128, num_actions),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		return self.head(x)