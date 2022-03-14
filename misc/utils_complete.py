from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
#from torch.utils.tensorboard import SummaryWriter

import numpy as np


def sample_softmax_with_temperature(logits, temperature=1):
    """
    Input:
    logits: Tensor of log probs, shape = BS x k
    temperature = scalar

    Output: Tensor of values sampled from Gumbel softmax.
            These will tend towards a one-hot representation in the limit of temp -> 0
            shape = BS x k
    """
    h = (logits) * temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y


def compute_pairwise_distance(x):
    ''' computation of pairwise distance matrix
    ---- Input
    - x: input tensor		torch.Tensor [(bs), sample_num, dim_x]
    ---- Return
    - matrix: output matrix	torch.Tensor [(bs), sample_num, sample_num]
    '''
    if len(x.shape) == 2:
        matrix = torch.norm(x[:, None, :] - x[None, :, :], p=2, dim=2)
    elif len(x.shape) == 3:
        matrix = torch.norm(x[:, :, None, :] - x[:, None, :, :], p=2, dim=3)
    else:
        raise NotImplementedError
    return matrix


def compute_norm_pairwise_distance(x):
    ''' computation of normalized pairwise distance matrix
    ---- Input
    - x: input tensor		torch.Tensor [(bs), sample_num, dim_x]
    ---- Return
    - matrix: output matrix	torch.Tensor [(bs), sample_num, sample_num]
    '''
    x_pair_dist = compute_pairwise_distance(x)
    normalizer = torch.sum(x_pair_dist, dim=-1)
    x_norm_pair_dist = x_pair_dist / (normalizer[..., None] + 1e-12).detach()
    return x_norm_pair_dist


def NDiv_loss(z, y, alpha=0.8):
    ''' NDiv loss function.
    ---- Input
    - z: latent samples after embedding h_Z:		torch.Tensor [(bs), sample_num, dim_z].
    - y: corresponding outputs after embedding h_Y:	torch.Tensor [(bs), sample_num, dim_y].
    - alpha: hyperparameter alpha in NDiv loss.
    ---- Return
    - loss: normalized diversity loss.			torch.Tensor [(bs)]
    '''
    S = z.shape[-2]  # sample number
    y_norm_pair_dist = compute_norm_pairwise_distance(y)
    z_norm_pair_dist = compute_norm_pairwise_distance(z)
    # ndiv_loss_matrix = F.relu(z_norm_pair_dist * alpha - y_norm_pair_dist)
    ndiv_loss_matrix = y_norm_pair_dist / z_norm_pair_dist
    
    eps = 1 * 1e-5
    ndiv_loss_matrix = 1 / (ndiv_loss_matrix + eps)
    ndiv_loss = ndiv_loss_matrix.sum(-1).sum(-1) / (S * (S - 1))
    return ndiv_loss

# Define Gumbel Function Utility
def sample_gumbel(shape, eps=1e-20):
    unif = torch.rand(*shape) # rand is uniform distribution by default;
    g = -torch.log(-torch.log(unif + eps)) # Double exponential function to become gumbel noise;
    return g

# Define Gumbel Sampling Strategy Utility
def sample_gumbel_softmax(logits, temperature):
    """
    Input:
    logits: Tensor of log probs, shape = BS x k
    temperature = scalar

    Output: Tensor of values sampled from Gumbel softmax.
            These will tend towards a one-hot representation 
            in the limit of temp -> 0
            shape = BS x k
    """
    g = sample_gumbel(logits.shape).cuda()
    h = (g + logits) * temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y


def custom_cross_entropy(input, target):
    return torch.mean(-torch.sum(target * torch.log(input), 1))


def custom_cross_entropy_v2(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-soft_targets * logsoftmax(pred), 1))


def custom_kldiv(pred, soft_targets):
    logsoftmax = nn.LogSoftmax()
    pred = logsoftmax(pred)
    target = torch.nn.functional.softmax(soft_targets, dim=1)
    return torch.nn.KLDivLoss(size_average=False)(pred, target)


def collate_func(data):
    return data


def get_transition_matrix(dataset, ncls):
    """get transition matrix to use in viterbi
    Input:
        dataset: (the lmdb dataset obtained from our loader)
        ncls: number of classes in your dataset
    """

    # init transition matrix
    trans = torch.zeros((ncls, ncls), dtype=torch.float32)
    # loop through entire dataset
    for sample in dataset:
        step_ids = sample["W"]
        # find neighboring pairs
        adj = tuple(zip(step_ids[:-1], step_ids[1:]))
        # update transition matrix
        for i in range(len(adj)):
            trans[adj[i]] += 1

    # normalize transition matrix
    tnorm = trans.sum(1).unsqueeze(1).repeat(1, ncls)
    trans = trans / tnorm

    return trans


def viterbi_path(
    transmat, emission, observations=None, prior=None, scaled=False, ret_loglik=False
):
    """Finds the most-probable viterbi path
    Inputs:
        transmat: np.ndarray((ncls,ncls))
        emission: np.ndarray((ncls,prediction_horizon))
        observations: ID of the observation (default to np.arange(prediction_horizon))
        scaled: bool
            whether or not to normalize the probability to prevents underflow
            by repeated multiplications of probabilities
        ret_loglik: bool
            whether or not to return the log-likelihood of the best path
    Outputs:
        path: np.array(prediction_horizon)
    """
    # get num steops
    num_steps = emission.shape[0]
    # get obserbvations (default to length of emission)
    if observations is None:
        observations = np.arange(emission.shape[1])

    num_obs = observations.shape[0]
    # get prior state probs
    if prior is None:
        prior = np.ones((num_steps,), dtype=np.float32) / num_steps

    # init viterbi
    T1 = np.zeros((num_steps, num_obs))

    T2 = np.zeros(
        (num_steps, num_obs), dtype=int
    )  # int because its elements will be used as indicies
    path = np.zeros(
        num_obs, dtype=int
    )  # int because its elements will be used as indicies

    T1[:, 0] = prior * emission[:, observations[0]]  # element-wise mult

    if scaled:
        scale = np.ones(num_obs)
        scale[0] = 1.0 / np.sum(T1[:, 0])
        T1[:, 0] *= scale[0]
    # go through viterbi rec
    T2[:, 0] = 0

    for t in range(1, num_obs):
        for j in range(num_steps):
            trans_probs = T1[:, t - 1] * transmat[:, j]
            # trans_probs = T1[
            # :, t - 1
            # ]  # I need to uniform-ify this transmat..., but how?
            T2[j, t] = trans_probs.argmax()
            T1[j, t] = trans_probs[T2[j, t]]
            T1[j, t] = 1.0
            T1[j, t] *= emission[j, observations[t]]
        if scaled:
            scale[t] = 1.0 / np.sum(T1[:, t])
            T1[:, t] *= scale[t]

    # unroll path
    path[-1] = T1[:, -1].argmax()
    for t in range(num_obs - 2, -1, -1):
        path[t] = T2[(path[t + 1]), t + 1]

    if not ret_loglik:
        return path
    else:
        if scaled:
            loglik = -np.sum(np.log(scale))
        else:
            p = T1[path[-1], -1]
            loglik = np.log(p)
        return path, loglik
