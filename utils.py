from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import nltk

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


def custom_KLDiv(input, target):
    return torch.mean(-torch.sum(target * torch.log(input), 1))

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

def onehot(labels: torch.Tensor, label_num):
    return torch.zeros(labels.shape[0], label_num, device=labels.device).scatter_(
        1, labels.view(-1, 1), 1
    )

class MILNCELoss_V3(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss_V3, self).__init__()
        self.ce_loss = torch.nn.CrossEntropyLoss()

    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0])
        label = torch.eye(x.shape[0])[:, :, None].cuda().long().squeeze().argmax(1)

        return self.ce_loss(x, label)

class MILNCELoss_V2(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss_V2, self).__init__()

    def forward(self, sim_matrix, sim_matrix_y, label):
        # x = torch.matmul(video_embd, text_embd.t())
        # x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        x = sim_matrix
        y = sim_matrix_y
        # nominator = x * torch.eye(x.shape[0])[:, :, None].cuda()
        label_vec = onehot(label, 105).cuda()

        nominator = torch.matmul(x, label_vec.t())
        # nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, y.t()), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)

class MILNCELoss(torch.nn.Module):
    def __init__(self):
        super(MILNCELoss, self).__init__()

    def forward(self, video_embd, text_embd):
        x = torch.matmul(video_embd, text_embd.t())
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)
        nominator = x * torch.eye(x.shape[0])[:, :, None].cuda()
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        return torch.mean(denominator - nominator)

def bleu(input, reference):
    """
    Example reference: reference1 = 'the cat is on the mat'.split()
    Example reference: reference2 = 'there is a cat on the mat'.split()

    Example input: hypothesis1 = 'the the the the the the the'.split()
    """

    reference1 = "a b c d".split()
    reference2 = "a c b d".split()

    input = "a b c e".split()
    input = "a c b e".split()
    ref = [reference1, reference2]

    print(nltk.translate.bleu_score.modified_precision(ref, input, n=4))


def sort_tuple(tup):
    tup.sort(key=lambda x: x[0])
    return tup


def entropy_reg(vector):
    """
    vector: normalized torch tensor that sum to 1.0
    """
    return -(torch.log(vector) * vector).mean()


def sort_tuple_batch(tup, batch=256):
    """What is the best way to do this?"""
    for b in range(batch):
        tmp = [x[0][b] for x in tup]
    return tup


def strictly_increasing(L):
    return all(x < y for x, y in zip(L, L[1:]))


def strictly_decreasing(L):
    return all(x > y for x, y in zip(L, L[1:]))


def non_increasing(L):
    return all(x >= y for x, y in zip(L, L[1:]))


def non_decreasing(L):
    return all(x <= y for x, y in zip(L, L[1:]))


def RankLoss(*input, margin=0):
    """Arbitrary length of input variables from start to end
    visual_emb: A list of [start, ..., goal] visual embeddings
    """
    visual_emb = input[0]
    goal = visual_emb[-1]
    loss = []

    """Loop through every pair of adjacent visual features, calc their similarity to goal and rank with margin"""
    for feat1, feat2 in zip(visual_emb[:-2], visual_emb[1:-1]):
        tmp = torch.norm(goal - feat2) - torch.norm(goal - feat1)
        if tmp > margin:
            loss.append(tmp - margin)
        else:
            loss.append(0)
    return sum(loss) / len(loss)


def checkRank(*input):
    """Arbitrary length of input variables from start to end"""
    visual_emb = input[0]
    goal = visual_emb[-1]
    dist = []
    for feat in visual_emb[:-1]:
        dist.append(torch.norm(goal.float() - feat.float()).item())

    rst = non_increasing(dist)  # Higher the better
    if rst:
        return 1.0
    else:
        return 0.0


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

