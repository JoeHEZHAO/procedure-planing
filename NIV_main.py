from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import collections

from eval_util import *
from utils import *
from layers import *

from models.model_NIV import *
from datasets.NIV_dataloader import *
from datasets.NIV_args import parse_args
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {} for experiment".format(device))

def collate_func(data):
    return data

args = parse_args()
args.transformer = False
args.batch_size = 32
args.pred_horz = 3
args.dataset = "niv"
args.dataloader_type = "ddn"
args.label_type = "ddn"
args.modeltype = "transformer"
args.d_model = 128
args.nlayer = 2
args.nhead = 8
args.exist_datasplit = True
print(args)

def temperature_softmax(logits, temperature):
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
            T2[j, t] = trans_probs.argmax()
            T1[j, t] = trans_probs[T2[j, t]]
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

with open(os.path.join(args.data_path, "action_ids.pickle"), "rb") as f:
    action_step_dict = pickle.load(f)

all_vids = os.listdir(os.path.join(args.data_path, "csvs"))
n_train = int(len(all_vids) * 0.7)
train_vids = np.random.choice(all_vids, n_train, replace=False).tolist()
test_vids = [vid for vid in all_vids if vid not in train_vids]

trainset = NIVDataset(
    task_vids=train_vids,
    n_steps=5,
    features_path=os.path.join(args.data_path, "processed_data"),
    constraints_path=os.path.join(args.data_path, "csvs"),
    step_cls_json=action_step_dict,
    pred_h=args.pred_horz,
)

testset = NIVDataset(
    task_vids=test_vids,
    n_steps=5,
    features_path=os.path.join(args.data_path, "processed_data"),
    constraints_path=os.path.join(args.data_path, "csvs"),
    step_cls_json=action_step_dict,
    pred_h=args.pred_horz,
)

##################################################
#  Loading existing data-split to reproduce      #
##################################################
if not args.exist_datasplit:
    datasplit = {}
    datasplit["train"] = trainset.plan_vids
    datasplit["test"] = testset.plan_vids
    with open("checkpoints/{}_t{}_datasplit.pth".format(args.dataset, args.pred_horz), "wb") as f:
        pickle.dump(datasplit, f)
else:
    with open("checkpoints/{}_t{}_datasplit.pth".format(args.dataset, args.pred_horz), "rb") as f:
        datasplit = pickle.load(f)
    trainset.plan_vids = datasplit["train"]
    testset.plan_vids = datasplit["test"]
print("length of testset {}".format(len(testset)))
print("length of trainset {}".format(len(trainset)))

########################################################
#  Viterbi Decoding Transition Matrix & Pre-processing #
########################################################
transition_matrix = get_transition_matrix(trainset, 48)

for i in range(transition_matrix.shape[1]):
    transition_matrix[:, i] = temperature_softmax(
        torch.where(
            transition_matrix[:, i] == 0.0,
            torch.finfo(torch.float64).eps * 100,  # 1e-7
            transition_matrix[:, i].double(),
        ),
        temperature=1,
    )

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

################################
#   Load NIV Lang Embeddings   #
################################
with open(os.path.join(args.data_path, "niv_act_embeddings.pickle"), "rb") as f:
    act_lang_emb = pickle.load(f)
act_lang_emb_sorted = collections.OrderedDict(sorted(act_lang_emb.items()))
act_tensor_list = list(act_lang_emb_sorted.values())
act_tensor_emb = torch.stack([torch.from_numpy(x) for x in act_tensor_list]).cuda()

vis_emb_dim, act_emb_dim, act_size, hidden_size = 512, 128, 48, 128
device = "cuda"

model = ProcedureFormer(
    input_dim=vis_emb_dim,
    d_model=args.d_model,
    cls_size=act_size,
    device="cuda",
    pred_horz=args.pred_horz,
    nhead=args.nhead,
    nlayer=args.nlayer,
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 5, gamma=0.7, last_epoch=-1)

#######################################################################################
#    Empirically decided Weighted CrossEntropy Loss for long-tailed distribution      #
#######################################################################################
ce_weight = torch.ones(48).cuda()
ce_inds = [26]
ce_weight[ce_inds] = 0.75
mse_loss = torch.nn.MSELoss()
ce_loss = th.nn.CrossEntropyLoss()
contra_loss = th.nn.CrossEntropyLoss()

def training(epoch):
    model.train()

    for batch in trainloader:
        optimizer.zero_grad()
        ce_loss_list = []
        state_pred_list = []
        label_list = []
        label_onehot_list = []

        for sample in batch:
            x = sample["X"].cuda().unsqueeze(
                0) if args.use_gpu else sample["X"]
            x = x.to(device)

            w = sample["W"].cuda().unsqueeze(
                0) if args.use_gpu else sample["W"]

            x = x.float()
            w = w.float()
            w = w.to(device)
            start_token = th.zeros(1).unsqueeze(-1).cuda()

            w = th.cat([start_token, w], 1).long()
            logits, state = model(x, args.pred_horz)

            state_pred_list.append(state)
            ce_loss_list.append(
                ce_loss(logits.squeeze(),
                        w[:, 1:].squeeze().reshape(-1, 1).squeeze())
            )
            label_list.append(w[:, 1:])

            """Make onehot label"""
            y_onehot_tmp = torch.FloatTensor(
                args.pred_horz, act_size).cuda()
            y_onehot_tmp.zero_()
            y_onehot_tmp.scatter_(1, (w[:, 1:]).view(args.pred_horz, -1), 1)
            label_onehot_list.append(y_onehot_tmp)

        "Language Contrastive Learning "
        act_tensor_enc = model.lang_encoder(act_tensor_emb)
        pred_state_enc = torch.stack(state_pred_list)
        labels = torch.stack(label_list).view(-1).squeeze()
        norm_pred = pred_state_enc.view(-1, args.d_model)
        norm_gt = act_tensor_enc
        pred_gt_sim = torch.matmul(norm_pred, norm_gt.T) * math.exp(0.7)
        contrastive_loss = contra_loss(pred_gt_sim, labels)

        ce_loss_avg = sum(ce_loss_list) / len(ce_loss_list)

        loss = ce_loss_avg + 0.5 * contrastive_loss
        loss.backward()
        optimizer.step()

def evaluation(epoch, model_path=False):
    gt_list = []
    pred_list = []
    pred_list_argmax = []
    logits_list = []
    if model_path:
        model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        for batch in testloader:
            for sample in batch:
                x = sample["X"].cuda().unsqueeze(
                    0) if args.use_gpu else sample["X"]
                x = x.to(device)

                w = sample["W"].cuda().unsqueeze(
                    0) if args.use_gpu else sample["W"]

                x = x.float()
                w = w.float()
                w = w.to(device)
                gt_list.append(w)

                start_token = (
                    th.zeros(1).unsqueeze(-1).float()
                )  # no gradient for this token;
                start_token = start_token.to(device)

                w = th.cat([start_token, w], 1).long()
                logits, _ = model(x, args.pred_horz)

                """Evaluate using viterbi-algorithm """
                path = viterbi_path(
                    transition_matrix.numpy(),
                    logits.squeeze().permute(1, 0).cpu().numpy(),
                )
                pred_list.append(torch.from_numpy(path).cuda())

                """ Evaluate using argmax algorithm """
                pred_list_argmax.append(logits.squeeze())

                """Save probabilitic logits """
                logits_list.append(logits.squeeze())
    """ Evaluate using argmax algorithm """
    rst_logits = torch.stack(logits_list)

    """ Evaluate using argmax algorithm """
    rst_argmax = torch.stack(pred_list_argmax)
    rst_argmax = torch.argmax(rst_argmax.view(-1, act_size), 1)
    rst_argmax = rst_argmax.view(-1, args.pred_horz)

    """ Evaluate using viterbi-algorithm """
    rst = torch.stack(pred_list)
    rst = rst.view(-1, args.pred_horz)

    gt = torch.stack(gt_list).squeeze().cpu().numpy().astype("int")
    rst = rst.cpu().numpy()
    rst_argmax = rst_argmax.cpu().numpy()

    sr = success_rate(rst, gt, False)
    sr_index = np.argwhere(sr == 1)
    rst_success = rst[sr_index].tolist()

    miou = acc_iou(rst, gt, False)
    macc = mean_category_acc(rst.flatten().tolist(), gt.flatten().tolist())

    print(
        "For epoch {} using viterbi-algorithm, Best Success Rate {}, meanIOU {} and meanACC {}".format(
            epoch, sr.mean(), miou.mean(), macc
        )
    )
    sr_argmax = success_rate(rst_argmax, gt, False)
    sr_index = np.argwhere(sr_argmax == 1)
    rst_success = rst[sr_index].tolist()

    miou = acc_iou(rst_argmax, gt, False)
    macc = mean_category_acc(
        rst_argmax.flatten().tolist(), gt.flatten().tolist())

    print(
        "For epoch {} using argmax, Best Success Rate {}, meanIOU {} and meanACC {}".format(
            epoch, sr_argmax.mean(), miou.mean(), macc
        )
    )

if __name__ == "__main__":
    train = False

    if train:
        for i in range(200):
            training(i)
            scheduler.step()
            evaluation(i)
            torch.save(
                model.state_dict(),
                "./result_{}_{}_{}_{}_{}/epoch_{}.pth".format(
                    args.dataset,
                    args.modeltype,
                    args.dataloader_type,
                    args.pred_horz,
                    args.label_type,
                    i,
                ),
            )
    else:
        model_path = "checkpoints/{}_best.pth".format(args.dataset)
        if model_path:
            model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        evaluation(0)