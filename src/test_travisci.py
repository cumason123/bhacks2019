import torch

def test_torch():
	x = torch.tensor(5)
	assert(x == torch.tensor(5))