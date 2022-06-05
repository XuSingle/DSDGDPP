from torch import distributions as dis
import numpy as np, itertools, collections, os, random, math
import matplotlib.pyplot as plt
import torch

def evaluate_samples(generated_samples, data, model_num, iteration, is_ring_distribution=True):
    generated_samples = generated_samples[:2500]
    data = data[:2500]

    if model_num % 2 == 0:
        model = 'Ring'
        thetas = np.linspace(0, 2 * np.pi, 8 + 1)[:-1]
        xs, ys = np.sin(thetas), np.cos(thetas)
        MEANS = np.stack([xs, ys]).transpose()
        std = 0.01
    else:
        model = 'Grid'
        MEANS = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                         range(-4, 5, 2))], dtype=np.float32)
        std = 0.05

    l2_store = []
    for x_ in generated_samples:
        l2_store.append([np.sum((x_ - i) ** 2) for i in MEANS])

    mode = np.argmin(l2_store, 1).flatten().tolist()
    dis_ = [l2_store[j][i] for j, i in enumerate(mode)]
    mode_counter = [mode[i] for i in range(len(mode)) if np.sqrt(dis_[i]) <= (3 * std)]

    #     sns.set(font_scale=2)
    #     f, (ax1, ax2) = plt.subplots(2, figsize=(10, 15))
    #     cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    #     sns.kdeplot(generated_samples[:, 0], generated_samples[:, 1], cmap=cmap, ax=ax1, n_levels=100, shade=True,
    #                 clip=[[-6, 6]] * 2)
    #     sns.kdeplot(data[:, 0], data[:, 1], cmap=cmap, ax=ax2, n_levels=100, shade=True, clip=[[-6, 6]] * 2)

    plt.figure(figsize=(5, 5))
    plt.scatter(generated_samples[:, 0], generated_samples[:, 1], edgecolor='none')
    plt.scatter(data[:, 0], data[:, 1], c='g', edgecolor='none')
    plt.axis('off')
    plt.savefig('Plots/%s_%s_iteration_%d.png' % ('GDPPGAN', model, iteration))
    # plt.show()
    plt.clf()
    # print(type(sum(list(np.sum(collections.Counter(mode_counter).values())))))
    # print(np.sum(collections.Counter(mode_counter).values()))
    # print(sum(list(np.sum(collections.Counter(mode_counter).values()))))
    high_quality_ratio = sum(list(np.sum(collections.Counter(mode_counter).values()))) / float(2500)
    print('Model: %d || Number of Modes Captured: %d' % (model_num, len(collections.Counter(mode_counter))))
    print('Percentage of Points Falling Within 3 std. of the Nearest Mode %f' % high_quality_ratio)

    return len(collections.Counter(mode_counter)), high_quality_ratio


def sample_ring(batch_size, n_mixture=8, std=0.01, radius=1.0):
    """Gnerate 2D Ring"""
#     std = [std, std]
#     std = torch.tensor(std)
    thetas = np.linspace(0, 2 * np.pi, n_mixture + 1)[:-1]
    xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
#     cat = dis.categorical.Categorical(logits = torch.zeros(n_mixture))
    cat = dis.categorical.Categorical(logits = torch.zeros(n_mixture))

    mean = torch.transpose(torch.tensor([xs.ravel(), ys.ravel()]), 0, 1)
#     print(mean)
    null = torch.empty(n_mixture, 2)
    std = torch.zeros_like(null) + std
    comps = dis.Independent(dis.normal.Normal(mean, std),1)
    data = dis.MixtureSameFamily(cat, comps)
    return data.sample(sample_shape=torch.tensor([batch_size]))


def sample_grid(batch_size, num_components=25, std=0.05):
    """Generate 2D Grid"""
    cat = dis.categorical.Categorical(logits = torch.zeros(num_components))
    mus = np.array([np.array([i, j]) for i, j in itertools.product(range(-4, 5, 2),
                                                                   range(-4, 5, 2))], dtype=np.float32)
    mean = torch.tensor(mus)
    null = torch.empty(num_components, 2)
    std = torch.zeros_like(null) + std
    comps = dis.Independent(dis.normal.Normal(mean, std),1)
    data = dis.MixtureSameFamily(cat, comps)
    return data.sample(sample_shape=torch.tensor([batch_size]))

def initialize_weights(net):
    for m in net.modules():
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
