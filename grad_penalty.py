import torch
import torch.nn as nn

def gradient_penalty(critic, real,fake, device):
    BATCH_SIZE, CH, D, H, W, = real.shape
    epsilon = torch.rand(BATCH_SIZE, 1,1,1,1).repeat(1,CH,D,H,W).to(device)
    interpolated_images = real * epsilon + fake * (1-epsilon)

    mixed_scores = critic(interpolated_images)
    # calculated scores
    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)

    return gradient_penalty
