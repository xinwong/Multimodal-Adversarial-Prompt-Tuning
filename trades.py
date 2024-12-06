import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0, distance='l_inf'):
    """
    TRADES loss function for adversarial training.

    Parameters:
    - model: The neural network model.
    - x_natural: The original input images.
    - y: The ground truth labels.
    - optimizer: The optimizer used for training.
    - step_size: The step size for the adversarial attack.
    - epsilon: The maximum perturbation allowed.
    - perturb_steps: The number of steps for generating adversarial examples.
    - beta: The weight for the robustness loss term.

    Returns:
    - loss: The combined loss of natural and adversarial examples.
    """
    # Define KL-divergence loss
    criterion_kl = torch.nn.KLDivLoss(reduction='batchmean')
    model.eval()
    batch_size = len(x_natural)

    # generate adversarial example
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
    
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv, attack="trades"), dim=1),
                                       F.softmax(model(x_natural, attack="trades"), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).cuda().detach()
        delta = Variable(delta.data, requires_grad=True)

        # Setup optimizers
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta

            # optimize
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * criterion_kl(F.log_softmax(model(adv), dim=1),
                                           F.softmax(model(x_natural), dim=1))
            loss.backward()
            # renorming gradient
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            # avoid nan or inf if gradient is 0
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0])
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        raise ValueError("Invalid distance metric. Use 'l_inf' or 'l_2'.")

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    model.train()

    # zero gradient
    optimizer.zero_grad()

    # calculate robust loss
    logits_natural = model(x_natural, attack="trades")
    logits_adv = model(x_adv, attack="trades")

    loss_natural = F.cross_entropy(logits_natural, y)
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits_natural, dim=1))
    loss = loss_natural + beta * loss_robust

    return loss

