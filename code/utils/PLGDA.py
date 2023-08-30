import numpy as np
import torch
import torch.nn.functional as F
import random


def augmentation(volume, aug_factor):
    # volume is numpy array of shape (C, D, H, W)
    return volume + aug_factor * np.clip(np.random.randn(*volume.shape) * 0.1, -0.2, 0.2).astype(np.float32)


def PLGDA_2D(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):

    X_b = len(X)
    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]

    U_b = len(U)
    U_cap = U.repeat(K, 1, 1, 1)
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)

    with torch.no_grad():
        a, b = eval_net(U_cap)
        Y_u = (a + b) / 2
        Y_u = F.softmax(Y_u, dim=1)

    guessed = torch.zeros(U.shape).repeat(1, 4, 1, 1)

    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K

    guessed = guessed.repeat(K, 1, 1, 1)
    guessed = torch.argmax(guessed, dim=1)
    pseudo_label = guessed

    guessed = guessed.detach().cpu().numpy()
    U_cap = U_cap.detach().cpu().numpy()
    U_cap = list(zip(U_cap, guessed))


    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'x':  # 保证取的范围在带标签的batch_size数的范围内, 取不带标签的batch_size数个进行融合
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')

    return X_prime, U_prime, pseudo_label


def PLGDA_3D(X, U, eval_net, K, T, alpha, mixup_mode, aug_factor):

    X_b = len(X)
    X_cap = [(augmentation(x[0], aug_factor), x[1]) for x in X]

    U_b = len(U)
    U_cap = U.repeat(K, 1, 1, 1, 1)
    U_cap += torch.clamp(torch.randn_like(U_cap) * 0.1, -0.2, 0.2)


    with torch.no_grad():
        a, b = eval_net(U_cap)
        Y_u = (a + b) / 2
        Y_u = F.softmax(Y_u, dim=1)

    guessed = torch.zeros(U.shape).repeat(1, 2, 1, 1, 1)
    guessed = guessed.cuda()
    for i in range(K):
        guessed += Y_u[i * U_b:(i + 1) * U_b]
    guessed /= K
    guessed = guessed.repeat(K, 1, 1, 1, 1)
    guessed = torch.argmax(guessed, dim=1)
    pseudo_label = guessed


    guessed = guessed.detach().cpu().numpy()
    U_cap = U_cap.detach().cpu().numpy()
    U_cap = list(zip(U_cap, guessed))


    x_mixup_mode, u_mixup_mode = mixup_mode[0], mixup_mode[1]

    if x_mixup_mode == '_':
        X_prime = X_cap
    else:
        raise ValueError('wrong mixup_mode')

    if u_mixup_mode == 'x':
        idxs = np.random.permutation(range(U_b * K)) % X_b
        U_prime = [mix_up(U_cap[i], X_cap[idxs[i]], alpha) for i in range(U_b * K)]
    elif u_mixup_mode == '_':
        U_prime = U_cap
    else:
        raise ValueError('wrong mixup_mode')

    return X_prime, U_prime, pseudo_label


def mix_up(s1, s2, alpha):

    l = np.random.beta(alpha, alpha)
    l = max(l, 1 - l)

    x1, p1 = s1
    x2, p2 = s2

    x = l * x1 + (1 - l) * x2
    p = l * p1 + (1 - l) * p2

    return (x, p)
