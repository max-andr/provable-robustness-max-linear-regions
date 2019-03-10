import numpy as np

import torch
from torch.autograd import Variable

from kolter_wong.convex_adversarial import DualNetBounds


def eval_lb_db(p_norm, model, loader, n_batches, device, alpha_init=1.0, epsilon_init=0.01, niters=20, threshold=1e-4):
    q_norm = {2: 'l2', np.inf: 'l1'}[p_norm]
    pred_correct, lbs = [], []

    # torch.save({'state_dict': [m.state_dict() for m in model]}, "mmr_at_fc1.pth")

    n_out = model[-1].out_features

    for j, (X, y) in enumerate(loader):
        if j == n_batches:
            break

        epsilon = Variable(epsilon_init * torch.ones(X.size(0)).to(device), requires_grad=True)
        X, y = Variable(X).to(device), Variable(y).to(device)

        out = Variable(model(X).data.max(1)[1])

        # form c without the 0 row
        # 20 x 10 x 10
        c = Variable(torch.eye(n_out).to(device)[out.data].unsqueeze(1) - torch.eye(n_out).to(device).unsqueeze(0)).to(device)
        # 20 x 10 x 1;   20 x 1 == 1 x 10  =>  20 x 10
        I = (~(out.data.unsqueeze(1) == torch.arange(n_out).to(device).unsqueeze(0)).unsqueeze(2))

        c = (c[I.expand_as(c)].view(X.size(0), n_out - 1, n_out))
        if X.is_cuda:
            c = c.to(device)

        def f(eps):
            dual = DualNetBounds(model, X, eps.unsqueeze(1), q_norm)
            f = -dual.g(c)
            return f.max(1)[0]

        for i in range(niters):
            f_max = f(epsilon)
            # if done, stop
            if (f_max.data.abs() <= threshold).all():
                break

            # otherwise, compute gradient and update
            f_max.sum().backward()

            alpha = alpha_init
            epsilon0 = Variable((epsilon - alpha * (f_max / epsilon.grad)).data,
                                requires_grad=True)

            while f(epsilon0).data.abs().sum() >= f_max.data.abs().sum():
                alpha *= 0.5
                epsilon0 = Variable((epsilon - alpha * (f_max / epsilon.grad)).data,
                                    requires_grad=True)
                if alpha <= 1e-3:
                    break

            epsilon = epsilon0
            del f_max

        lbs.append(epsilon)
        pred_correct.append(y == out)

        torch.cuda.empty_cache()  # to prevent undesirable memory growth
        del X, y

    lbs = torch.cat(lbs).cpu().detach().numpy()
    pred_correct = torch.cat(pred_correct).cpu().numpy().astype(bool)  # necessary

    return lbs, pred_correct

