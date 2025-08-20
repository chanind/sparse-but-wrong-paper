import torch


@torch.no_grad()
def random_model_params(model: torch.nn.Module):
    for param in model.parameters():
        param.data = torch.randn_like(param)
