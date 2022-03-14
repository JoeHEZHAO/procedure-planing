import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import nltk


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


if __name__ == "__main__":
    # test = [4, 3, 2, 6]
    # test_in = [torch.from_numpy(np.array(x)).float() for x in test]
    # print(test_in)
    # out = RankLoss(test_in)
    # print(out)
    # print(checkRank(test_in))

    # random = torch.rand(5)
    # print(entropy_reg(random))
    # random = torch.rand(5)
    # print(entropy_reg(random))
    nce_loss = MILNCELoss_V2()
    text_rand = torch.randn(2, 105).cuda()
    video_rand = torch.randn(105, 128).cuda()

    # label = torch.zeros(text_rand.shape[0]).cuda()
    # label[1] = 1
    label = [0, 1]
    label = np.array(label)
    label = torch.from_numpy(label)

    loss = nce_loss(text_rand, video_rand, label)

    bleu(0, 0)
