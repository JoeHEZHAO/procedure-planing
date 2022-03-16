from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from eval_util import *
from utils import *
from layers import *
from models.model_CrossTask import *
import pickle
import collections
from datasets.COIN_args import *
from datasets.COIN_dataloader import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {} for experiment".format(device))


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
    for idx, row in enumerate(trans):
        if (row == 0).all():
            trans[idx] = torch.ones(row.shape) * (1 / ncls)
    tnorm = trans.sum(1).unsqueeze(1).repeat(1, ncls)
    trans = trans / tnorm

    return trans

def sample_gumbel_softmax_v2(logits, temperature):
    """
    Input:
    logits: Tensor of log probs, shape = BS x k
    temperature = scalar

    Output: Tensor of values sampled from Gumbel softmax.
            These will tend towards a one-hot representation in the limit of temp -> 0
            shape = BS x k
    """
    h = (logits) * temperature
    # g = sample_gumbel(logits.shape).cuda()
    # h = (g + logits.cuda()) * temperature
    h_max = h.max(dim=-1, keepdim=True)[0]
    h = h - h_max
    cache = torch.exp(h)
    y = cache / cache.sum(dim=-1, keepdim=True)
    return y

####################################################################################
#    Mean and Variance for Visual & Language feature Estimated from Traing-set     #
####################################################################################
mean_lang = -0.005225768
mean_vis = 0.000133333
var_lang = 36.842754
var_vis = 0.00021489676

args = parse_args()
args.transformer = False
args.batch_size = 32
args.pred_horz = 3
args.dataset = "coin"
args.dataloader_type = "ddn"
args.label_type = "ddn"
args.modeltype = "transformer"
args.d_model = 128
args.nlayer = 2
args.nhead = 8
print(args)

"""Declaring the tensorboard to log the stats"""
if not os.path.exists(
    "output_logging/result_{}_{}_{}_{}_{}".format(
        args.dataset,
        args.modeltype,
        args.dataloader_type,
        args.pred_horz,
        args.label_type,
    )
):
    os.mkdir(
        "output_logging/result_{}_{}_{}_{}_{}".format(
            args.dataset,
            args.modeltype,
            args.dataloader_type,
            args.pred_horz,
            args.label_type,
        )
    )

""" Notice: random_split for 30/70"""
trainset = CoinTaskDataset(
    task_vids=os.path.join(args.data_path, "train_split.pickle"),
    n_steps=5,
    features_path=os.path.join(args.data_path, "full_npy"),
    constraints_path=os.path.join(args.data_path, "COIN.json"),
    step_cls_json=os.path.join(args.data_path, "steps_info.pickle"),
    pred_h=args.pred_horz,
)
trainset.mean_lan = mean_lang
trainset.mean_vis = mean_vis
trainset.var_lan = var_lang
trainset.var_vis = var_vis

testset = CoinTaskDataset(
    task_vids=os.path.join(args.data_path,"test_split.pickle"),
    n_steps=5,
    features_path=os.path.join(args.data_path,"full_npy"),
    constraints_path=os.path.join(args.data_path,"COIN.json"),
    step_cls_json=os.path.join(args.data_path,"steps_info.pickle"),
    pred_h=args.pred_horz,
    train=False
)
testset.mean_lan = mean_lang
testset.mean_vis = mean_vis
testset.var_lan = var_lang
testset.var_vis = var_vis

transition_matrix = get_transition_matrix(trainset, 778 + 1)[1:, 1:]

trainloader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=True,
    drop_last=True,
    collate_fn=collate_func,
)
testloader = DataLoader(
    testset,
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    drop_last=False,
    collate_fn=collate_func,
)
# Show stats of train/test dataset
print("Training dataset has {} samples".format(len(trainset)))
print("Testing dataset has {} samples".format(len(testset)))

vis_emb_dim, act_emb_dim, act_size, hidden_size = 512, 128, 778 + 1, 128
device = "cuda"

with open(os.path.join(args.data_path,"steps_info.pickle"), "rb") as f:
    act_data = pickle.load(f)
act_lang_emb = act_data["steps_to_embeddings"]

act_lang_emb_sorted = collections.OrderedDict(sorted(act_lang_emb.items()))
act_tensor_list = list(act_lang_emb_sorted.values())
act_tensor_emb = torch.stack(
    [torch.from_numpy(x) for x in act_tensor_list]
)  # [all_step_emb, feat_dim (512)]

model = ProcedureFormer(
    input_dim=vis_emb_dim,
    d_model=args.d_model,
    cls_size=act_size,
    device="cuda",
    pred_horz=args.pred_horz,
    nhead=args.nhead,
    nlayer=args.nlayer,
    noise_dim=32,
).to(device)

optimizer = optim.RMSprop(model.parameters(), lr=7e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 10, gamma=0.5, last_epoch=-1)

mse_loss = torch.nn.MSELoss()
ce_loss = th.nn.CrossEntropyLoss()

###############################################
# Normalize the Transition Matrix row-by-row  #
###############################################
for i in range(transition_matrix.shape[1]):
    transition_matrix[:, i] = sample_gumbel_softmax_v2(
        transition_matrix[:, i],
        temperature=1.0,
    )

def training(epoch):
    global args
    model.train()

    for batch in trainloader:
        optimizer.zero_grad()
        loss1, loss2 = [], []
        state_pred_list = []
        label_list = []
        label_onehot_list = []
        for sample in batch:
            x = sample["X"].cuda().unsqueeze(
                0) if args.use_gpu else sample["X"]
            x = x.to(device)

            w = sample["W"].cuda().unsqueeze(
                0) if args.use_gpu else sample["W"]

            x = x.unsqueeze(0).float()
            w = w.unsqueeze(0).float()
            w = w.to(device)
            start_token = th.zeros(1).unsqueeze(-1).cuda()
            w = th.cat([start_token, w], 1).long()

            logits, state, noise = model(x, args.pred_horz)
            gt_state = model.state_encoder(x).mean(2)

            state_pred_list.append(state)
            y_onehot_tmp = torch.FloatTensor(
                args.pred_horz, act_size - 1).cuda()
            y_onehot_tmp.zero_()
            y_onehot_tmp.scatter_(1, (w[:, 1:]-1).view(args.pred_horz, -1), 1)
            label_onehot_list.append(y_onehot_tmp)

            label_list.append(w[:, 1:])

            "Strong state supervision "
            loss1.append(mse_loss(state.squeeze(), gt_state[:, 1:]))

            "CE loss"
            loss2.append(
                ce_loss(logits.squeeze(),
                        w[:, 1:].squeeze().reshape(-1, 1).squeeze())
            )

        "Weak-Language Contrastive supervision "
        act_tensor_enc = model.lang_encoder(act_tensor_emb)
        pred_state_enc = torch.stack(state_pred_list)
        labels = torch.stack(label_list).view(-1).squeeze() - 1
        norm_pred = pred_state_enc.view(-1, args.d_model - args.noise_dim)
        norm_gt = act_tensor_enc
        pred_gt_sim = torch.matmul(norm_pred, norm_gt.T) * math.exp(0.7)
        contrastive_loss = ce_loss(pred_gt_sim, labels)

        # Note that loss1, state mse loss is not used
        cross_entropy_loss = sum(loss2) / len(loss2)
        loss = cross_entropy_loss + 0.5 * contrastive_loss

        loss.backward()
        optimizer.step()
    print("For batch {}, finish traning".format(i))


def evaluation(epoch, model_path=False):
    gt_list = []
    pred_list = []
    pred_list_argmax = []
    pred_entropy_list = []

    if model_path:
        model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            for sample in batch:
                x = sample["X"].cuda().unsqueeze(
                    0) if args.use_gpu else sample["X"]
                x = x.to(device)

                w = sample["W"].cuda().unsqueeze(
                    0) if args.use_gpu else sample["W"]
                w = w.to(device)

                x = x.float()
                w = w.float()
                gt = w
                gt_list.append(w)

                start_token = (
                    th.zeros(1).unsqueeze(-1).float()
                )  # no gradient for this token;
                start_token = start_token.to(device)

                w = th.cat([start_token, w], 1).long()

                logits, _, _ = model(x, args.pred_horz)

                norm_logits = F.softmax(logits.squeeze())
                entropy_loss = -(norm_logits * torch.log(norm_logits)).mean()
                if not torch.isnan(entropy_loss):
                    pred_entropy_list.append(entropy_loss)

                """ Evaluate using viterbi-algorithm """
                path = viterbi_path(
                    transition_matrix.numpy(),
                    norm_logits.squeeze().permute(1, 0)[1:].cpu().numpy(),

                )
                pred_list.append(torch.from_numpy(path).cuda())

                """ Evaluate using argmax algorithm """
                pred_list_argmax.append(logits.squeeze())

    """ Evaluate using argmax algorithm """
    rst_argmax = torch.stack(pred_list_argmax)
    rst_argmax = torch.argmax(rst_argmax.view(-1, act_size), 1)
    rst_argmax = rst_argmax.view(-1, args.pred_horz)

    """ Evaluate using viterbi-algorithm """
    rst = torch.stack(pred_list)
    rst = rst.view(-1, args.pred_horz)

    gt = torch.stack(gt_list).squeeze().cpu().numpy().astype("int")
    rst = rst.cpu().numpy() + 1
    rst_argmax = rst_argmax.cpu().numpy()

    sr = success_rate(rst, gt, False)
    miou = acc_iou(rst, gt, False)
    macc = mean_category_acc(rst.flatten().tolist(), gt.flatten().tolist())

    print(
        "For Horizon {}, epoch {} using viterbi-algorithm, Best Success Rate {}, meanIOU {}, meanACC {}".format(
            args.pred_horz,
            epoch,
            sr.mean(),
            miou.mean(),
            macc,
        )
    )

    sr = success_rate(rst_argmax, gt, False)
    miou = acc_iou(rst_argmax, gt, False)
    macc = mean_category_acc(
        rst_argmax.flatten().tolist(), gt.flatten().tolist())

    print(
        "For epoch {} using argmax, Best Success Rate {}, meanIOU {}, meanACC {}".format(
            epoch,
            sr.mean(),
            miou.mean(),
            macc
        )
    )


if __name__ == "__main__":
    # train = True
    train = False

    if train:
        for i in range(200):
            training(i)
            evaluation(i)

            torch.save(
                model.state_dict(),
                "output_logging/result_{}_{}_{}_{}_{}/epoch_{}.pth".format(
                    args.dataset,
                    args.modeltype,
                    args.dataloader_type,
                    args.pred_horz,
                    args.label_type,
                    i,
                ),
            )
        
            scheduler.step()
    else:
        model_path = os.path.join(
            'checkpoints',
            'COIN_best.pth'
        )
        print("Using weights model {}".format(model_path))
        evaluation(0, model_path)
