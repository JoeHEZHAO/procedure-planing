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
import json
import pickle

import collections
from datasets.CrossTask_args import parse_args
from datasets.CrossTask_dataloader import *

from eval_util import *
from utils import *
from layers import *
from models.model_CrossTask import *
from collections import Counter
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device {} for experiment".format(device))

##############################################################################################
# Mean/Variance value of visual feature as well as language feature Estimated from train-set #
##############################################################################################
mean_lang = 0.038948704
mean_vis = 0.000133333
var_lang = 33.063942
var_vis = 0.00021489676

args = parse_args()
args.spec_note = "completeLoss"
args.d_model = 128
args.noise_dim = 32
args.batch_size = 32
args.exist_datasplit = True
print("Using the following arguments for experiments: \n {}".format(args))

"""Declaring the tensorboard to log the stats"""
dir_path = "output_logging/result_{}_{}_{}_{}_{}_l{}h{}_{}".format(
    args.dataset,
    args.modeltype,
    args.dataloader_type,
    args.pred_horz,
    args.label_type,
    args.nlayer,
    args.nhead,
    args.spec_note,
)

########################################
# Start Loading/Processing the dataset #
########################################
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
task_vids = get_vids(args.video_csv_path)
val_vids = get_vids(args.val_csv_path)
task_vids = {
    task: [vid for vid in vids if task not in val_vids or vid not in val_vids[task]]
    for task, vids in task_vids.items()
}
primary_info = read_task_info(args.primary_path)
test_tasks = set(primary_info["steps"].keys())
if args.use_related:
    related_info = read_task_info(args.related_path)
    task_steps = {**primary_info["steps"], **related_info["steps"]}
    n_steps = {**primary_info["n_steps"], **related_info["n_steps"]}
else:
    task_steps = primary_info["steps"]
    n_steps = primary_info["n_steps"]
all_tasks = set(n_steps.keys())
task_vids = {task: vids for task,
             vids in task_vids.items() if task in all_tasks}
val_vids = {task: vids for task, vids in val_vids.items() if task in all_tasks}

with open(os.path.join(args.data_path, "crosstask_release/cls_step.json"), "r") as f:
    step_cls = json.load(f)
with open(os.path.join(args.data_path, "crosstask_release/activity_step.json"), "r") as f:
    act_cls = json.load(f)

##################################
# If using existing data-split   #
##################################
if args.exist_datasplit:
    with open("./checkpoints/CrossTask_t{}_datasplit.pth".format(args.pred_horz), "rb") as f:
        datasplit = pickle.load(f)
    trainset = CrossTaskDataset(
        task_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.pred_horz,
        act_json=act_cls,
    )
    testset = CrossTaskDataset(
        task_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.pred_horz,
        act_json=act_cls,
        train=False,
    )
    trainset.plan_vids = datasplit["train"]
    testset.plan_vids = datasplit["test"]

else:
    """ Random Split dataset by video """
    train_vids, test_vids = random_split(
        task_vids, test_tasks, args.n_train, seed=99999999)

    trainset = CrossTaskDataset(
        train_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.pred_horz,
        act_json=act_cls,
    )

    # Run random_split for eval/test sub-set
    # trainset.random_split()
    testset = CrossTaskDataset(
        test_vids,
        n_steps,
        args.features_path,
        args.annotation_path,
        step_cls,
        pred_h=args.pred_horz,
        act_json=act_cls,
        train=False,
    )

#######################
# Run data whitening  #
#######################
trainset.mean_lan = mean_lang
trainset.mean_vis = mean_vis
trainset.var_lan = var_lang
trainset.var_vis = var_vis
testset.mean_lan = mean_lang
testset.mean_vis = mean_vis
testset.var_lan = var_lang
testset.var_vis = var_vis

##################################################################
# Calculate the Transition Matrix for Viterbi Decoding Algorithm #
##################################################################
transition_matrix = get_transition_matrix(trainset, 106)[1:, 1:]
""" Normalize the Transition Matrix row-by-row """
for i in range(transition_matrix.shape[1]):
    transition_matrix[:, i] = sample_softmax_with_temperature(
        transition_matrix[:, i],
    )

#######################
# Init the DataLoader #
#######################
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

"""Get all reference from test-set, for KL-Divgence, NLL, MC-Prec and MC-Rec"""
reference = [x[2] for x in testset.plan_vids]
all_ref = np.array(reference)

##################################
# Saving the data split to local #
##################################
if not args.exist_datasplit:
    datasplit = {}
    datasplit["train"] = trainset.train_plan_vids
    datasplit["test"] = trainset.test_plan_vids
    with open("CrossTask_t{}_datasplit.pth".format(args.pred_horz), "wb") as f:
        pickle.dump(datasplit, f)

########################################
# Start Loading/Initializing the Model #
########################################
vis_emb_dim, act_emb_dim, act_size, hidden_size = 512 + 128, 128, 106, 128 # 512 (s3d) + 128 (vgg-audio)
model = ProcedureFormer(
    input_dim=vis_emb_dim,
    d_model=args.d_model,
    cls_size=act_size,
    device="cuda",
    pred_horz=args.pred_horz,
    nhead=args.nhead,
    nlayer=args.nlayer,
    noise_dim=args.noise_dim,
).to(device)

#######################
# Init the optimizers #
#######################
optimizer = optim.Adam(
    [
        {"params": model.state_encoder.parameters()},
        {"params": model.state_decoder.parameters()},
        {"params": model.lang_encoder.parameters()},
        {"params": model.query_embed.parameters()},
        {"params": model.keyvalue_embed.parameters()},
        {"params": model.tf_decoder.parameters()},
        {"params": model.pos_encoder.parameters()},
        {"params": model.cls_decoder.parameters()},
    ],
    lr=7e-4,
)
optimizer_d = optim.Adam(
    [
        {"params": model.discriminator_pred_cls_enc.parameters()},
        {"params": model.discriminator1.parameters()},
        {"params": model.discriminator2.parameters()},
    ],
    lr=1e-5,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, 20, gamma=0.65, last_epoch=-1)

#######################################################################################
#    Empirically decided Weighted CrossEntropy Loss for long-tailed distribution      #
#######################################################################################
ce_weight = torch.ones(106)
ce_inds = [1, 5, 37, 57, 35, 36]
ce_weight[ce_inds] = 0.75
ce_loss = th.nn.CrossEntropyLoss()

mse_loss = torch.nn.MSELoss()
nll_loss = nn.NLLLoss()
nce_loss = MILNCELoss_V2()
contra_loss = CrossEntropyLoss()
bce_loss = th.nn.BCELoss()

#########################################
#  Load pre-trained language embedding  #
#########################################
with open(os.path.join(
    args.data_path, "crosstask_release/act_lang_emb.pkl"), "rb") as f:
    act_lang_emb = pickle.load(f)
act_lang_emb_sorted = collections.OrderedDict(sorted(act_lang_emb.items()))
act_tensor_list = list(act_lang_emb_sorted.values())
act_tensor_emb = torch.stack([torch.from_numpy(x) for x in act_tensor_list]).cuda()

def train_complete_loss(epoch, NDR_train=True):
    """
    Train the model with complete loss function:
        1). CE loss for action labels;
        2). NCE loss for intermeidate states;
        3). Adv loss for both;
        4). NDR loss for both;
    """

    print("For epoch {}, start training the model-generator with complete loss".format(epoch))
    model.train()
    adv_d_N = 2

    for batch in trainloader:
        optimizer.zero_grad()
        optimizer_d.zero_grad()
        loss1, loss2 = [], []
        state_pred_list = []
        word_pred_list = []  # logits before differeniatable sampling
        label_list = []
        label_onehot_list = []
        loss_ndr = []

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

            # start = time.time()
            logits, state, _ = model(x, args.pred_horz)
            # print("Transformer running time is {}".format(time.time() - start))
            gt_state = model.state_encoder(x).mean(2)

            state_pred_list.append(state)
            word_pred_list.append(logits)

            """Make onehot label"""
            y_onehot_tmp = torch.FloatTensor(
                args.pred_horz, act_size - 1).cuda()
            y_onehot_tmp.zero_()
            y_onehot_tmp.scatter_(1, (w[:, 1:]-1).view(args.pred_horz, -1), 1)
            label_onehot_list.append(y_onehot_tmp)

            label_list.append(w[:, 1:])
            loss1.append(mse_loss(state.squeeze(), gt_state[:, 1:])) # Strong supervision is not used.
            loss2.append(
                ce_loss(logits.squeeze(),
                        w[:, 1:].squeeze().reshape(-1, 1).squeeze())
            )

            """Generate some other results for NDR """
            ####################
            #  Train with NDR  #
            ####################
            if NDR_train:
                num_ndr_sample = 20
                noise_z = (
                    torch.randn(num_ndr_sample,
                                args.noise_dim).cuda()
                ) * 1  # Standard normal noise multiply variance, {1, 2, 5, 10} all would work;
                "Repeat the noise for every horizon step:"
                noise_z = noise_z.reshape(
                    1, num_ndr_sample, args.noise_dim).repeat(x.shape[1], 1, 1)
                replicate_x = x.repeat(num_ndr_sample, 1, 1, 1)
                _, state_pred_samples = model.inference(
                    replicate_x, noise_z, args.pred_horz, NDR=True)
                # state_pred_samples = state_pred_samples.reshape(
                #     args.pred_horz, num_ndr_sample, -1).permute(1, 0, 2).reshape(1, num_ndr_sample, -1)
                state_1, state_2 = torch.unbind(
                    state_pred_samples.mean(0), dim=0)
                noise_z1, noise_z2 = torch.unbind(noise_z.reshape(
                    args.pred_horz+1, num_ndr_sample, -1).mean(0), dim=0)
                loss_ndr_tmp = torch.mean(
                    torch.abs(state_1 - state_2)) / torch.mean(torch.abs(noise_z1 - noise_z2))
                eps = 1 * 1e-5
                loss_ndr.append(1 / (eps + loss_ndr_tmp))
            else:
                loss_ndr.append(0.0)

        "Gumbel Sampling for Adv"
        pred_word_tensor = torch.stack(word_pred_list)
        pred_word_sampling = sample_gumbel_softmax(
            # Tune this temperature for better results;
            pred_word_tensor.reshape(-1, act_size), temperature=0.5
        )
        "Encode pred_word_sampling as well as the ground-truth word"
        pred_word_sampling_enc = torch.matmul(
            pred_word_sampling, model.discriminator_pred_cls_enc.weight)
        pred_word_sampling_enc = pred_word_sampling_enc.reshape(
            args.batch_size, args.pred_horz, -1)
        real_word_tensor = torch.cat([torch.zeros(torch.stack(label_onehot_list)[
                                     :, :, 0:1].shape).cuda(), torch.stack(label_onehot_list)], -1).cuda()

        real_word_enc = torch.matmul(
            real_word_tensor,
            model.discriminator_pred_cls_enc.weight
        )

        "Contrastive Learning "
        act_tensor_enc = model.lang_encoder(act_tensor_emb)
        pred_state_enc = torch.stack(state_pred_list)
        labels = torch.stack(label_list).view(-1).squeeze() - 1
        norm_pred = pred_state_enc.view(-1, args.d_model - args.noise_dim)
        norm_gt = act_tensor_enc
        pred_gt_sim = torch.matmul(norm_pred, norm_gt.T) * math.exp(0.7)
        pred_gt_sim_y = torch.matmul(norm_gt, norm_pred.T) * math.exp(0.7)

        """Define two fashion for mil-nce loss"""
        c_loss_v1 = contra_loss(pred_gt_sim, labels)

        "Adv learning for Generator "
        state_pred_tensor = torch.stack(state_pred_list).squeeze()
        label_onehot = torch.stack(label_onehot_list)

        state_real_tensor = torch.matmul(
            label_onehot.reshape(
                args.batch_size * args.pred_horz, -1), act_tensor_enc
        )
        state_real_tensor = state_real_tensor.reshape(
            args.batch_size, args.pred_horz, -1
        )

        g_fake_logits = torch.nn.functional.sigmoid(
            model.discriminator_forward(
                torch.cat([state_pred_tensor, pred_word_sampling_enc], -1)
            ).squeeze()
        )

        g_real_logits = torch.nn.functional.sigmoid(
            model.discriminator_forward(
                torch.cat([state_real_tensor, real_word_enc], -1)
            ).squeeze()
        )

        gt_real = torch.ones(g_real_logits.shape).cuda()
        gt_fake = torch.zeros(g_fake_logits.shape).cuda()

        adv_g_loss = bce_loss(g_real_logits, gt_real) + \
            bce_loss(g_fake_logits, gt_fake)

        # Loss for Generator
        loss2 = sum(loss2) / len(loss2)
        if NDR_train:
            loss = loss2 + 0.5 * c_loss_v1 + 0.1 * adv_g_loss + \
                0.1 * (sum(loss_ndr) / len(loss_ndr))
        else:
            loss = loss2 + 0.5 * c_loss_v1 + 0.1 * adv_g_loss

        if (epoch+1) % adv_d_N == 0:
            "Adv learning for Discriminator, with an interval of adv_d_N epoch "
            adv_d_loss = 0.5 * bce_loss(g_fake_logits, gt_real)
            adv_d_loss.backward()
            optimizer_d.step()
        else:
            "Only update Generator"
            loss.backward()
            optimizer.step()

    print("For batch {}, finish traning".format(i))
    print("For epoch {}, start training the model-generator with regular loss".format(epoch))
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
            x = x.cuda().float()

            w = sample["W"].cuda().unsqueeze(
                0) if args.use_gpu else sample["W"]
            w = w.cuda().float()

            start_token = th.zeros(1).unsqueeze(-1).cuda()

            w = th.cat([start_token, w], 1).long()

            logits, state, _ = model(x, args.pred_horz)
            gt_state = model.state_encoder(x).mean(2)

            state_pred_list.append(state)
            label_list.append(w[:, 1:])

            """Make onehot label"""
            y_onehot_tmp = torch.FloatTensor(
                args.pred_horz, act_size - 1).cuda()
            y_onehot_tmp.zero_()
            y_onehot_tmp.scatter_(1, (w[:, 1:]-1).view(args.pred_horz, -1), 1)
            label_onehot_list.append(y_onehot_tmp)

            loss1.append(mse_loss(state.squeeze(), gt_state[:, 1:]))
            loss2.append(
                ce_loss(logits.squeeze(),
                        w[:, 1:].squeeze().reshape(-1, 1).squeeze())
            )

        "Contrastive learning "
        act_tensor_enc = model.lang_encoder(act_tensor_emb)
        pred_state_enc = torch.stack(state_pred_list)
        labels = torch.stack(label_list).view(-1).squeeze() - 1 # minus 1 because action_id for CrossTask starts from 1;
        norm_pred = pred_state_enc.view(-1, args.d_model - args.noise_dim)
        norm_gt = act_tensor_enc
        # breakpoint()

        pred_gt_sim = torch.matmul(norm_pred, norm_gt.T) * math.exp(0.7)
        pred_gt_sim_y = torch.matmul(norm_gt, norm_pred.T) * math.exp(0.7)

        """Define two fashion for mil-nce loss"""
        c_loss_v1 = contra_loss(pred_gt_sim, labels)
        # c_loss_v2 = nce_loss(pred_gt_sim, pred_gt_sim_y, labels)

        # "Adv learning for Generator "
        # state_pred_tensor = torch.stack(state_pred_list).squeeze()
        # label_onehot = torch.stack(label_onehot_list)

        # state_real_tensor = torch.matmul(
        #     label_onehot.reshape(
        #         args.batch_size * args.pred_horz, -1), act_tensor_enc
        # )
        # state_real_tensor = state_real_tensor.reshape(
        #     args.batch_size, args.pred_horz, -1
        # )

        # g_fake_logits = torch.nn.functional.sigmoid(
        #     model.discriminator_forward(state_pred_tensor).squeeze()
        # )

        # g_real_logits = torch.nn.functional.sigmoid(
        #     model.discriminator_forward(state_real_tensor).squeeze()
        # )

        # gt_real = torch.ones(g_real_logits.shape).cuda()
        # gt_fake = torch.zeros(g_fake_logits.shape).cuda()

        # adv_loss = bce_loss(g_real_logits, gt_real) + \
        #     bce_loss(g_fake_logits, gt_fake)

        loss1 = 0.5 * sum(loss1) / len(loss1)
        loss2 = sum(loss2) / len(loss2)

        loss = loss2 + 0.5 * c_loss_v1
        loss.backward()
        optimizer.step()
    print("For batch {}, finish traning".format(i))

def inference(epoch, model_path=False, num_sampling=1500):
    global args
    gt_list = []
    pred_list = []
    pred_list_argmax = []
    pred_entropy_list = []
    ref_ce_list = []
    klv_list = []
    mc_prec = []
    mc_recall = []
    mode_rst = []
    nll_rst = []
    if model_path:
        model.load_state_dict(torch.load(model_path), strict=False)
        print("loading model weights from {}".format(model_path))
    # model.eval()

    with torch.no_grad():
        for idx, batch in enumerate(testloader):
            for sample in batch:
                x = sample["X"].cuda().unsqueeze(
                    0) if args.use_gpu else sample["X"]
                x = x.to(device).float()
                w = sample["W"].cuda().unsqueeze(
                    0) if args.use_gpu else sample["W"]
                w = w.to(device).float()

                gt = w
                gt_list.append(w)

                start_token = (
                    th.zeros(1).unsqueeze(-1).float()
                )
                start_token = start_token.to(device)

                w = th.cat([start_token, w], 1).long()

                """
                print("Sampling from noise only")
                for _ in range(num_sampling):
                    # noise with some variance
                    noise_z = torch.randn(
                        x.shape[1], x.shape[0], 32).cuda() * 10
                    logits, _ = model.inference(x, noise_z, args.pred_horz)
                    model_logits = logits.clone().squeeze()
                    rst_argmax = model_logits.argmax(1)
                    print(rst_argmax)
                """

                sample_listing = []
                sample_list = []
                ratio_list = []

                #########################################
                # Generate multiple samples in parallel #
                #########################################
                noise_z = (
                    torch.randn(1, num_sampling,
                                args.noise_dim).cuda()
                ) * 1.0  # Standard normal noise multiply variance, {1, 2, 5, 10} all would work;
                "Repeat the noise for every horizon step:"
                noise_z = noise_z.repeat(x.shape[1], 1, 1)

                with torch.no_grad():
                    replicate_x = x.repeat(num_sampling, 1, 1, 1)
                    logits, _ = model.inference(
                        replicate_x, noise_z, args.pred_horz)
                    model_logits = logits.clone().squeeze().cuda()

                # Comment out if using
                rst_argmax = model_logits.argmax(-1).permute(1, 0)
                sample_listing = rst_argmax

                "For mode eval"
                for vec in rst_argmax:
                    tmp = vec.squeeze().cpu().numpy().tolist()
                    tmp_str = "_".join([str(x) for x in tmp])
                    sample_list.append(tmp_str)

                # """Formulate distribution from these samples, for viterbi results """
                ref_onehot = torch.FloatTensor(args.pred_horz, act_size).cuda()
                ref_onehot.zero_()

                """Make this run in parallel"""
                ref_onehot_tmp = torch.FloatTensor(
                    rst_argmax.shape[0], args.pred_horz, act_size).cuda().zero_()
                vec_stack = sample_listing
                ref_onehot_tmp.scatter_(2, vec_stack.view(
                    rst_argmax.shape[0], args.pred_horz, -1), 1)
                ref_onehot = ref_onehot_tmp.sum(0)

                "Normalize with total number of samples"
                new_logits = ref_onehot / num_sampling

                #################
                #  Run Viterbi  #
                #################
                viterbi_rst = viterbi_path(
                    transition_matrix.numpy(),
                    new_logits.permute(1, 0)[1:].cpu().numpy()
                )
                pred_list.append(torch.from_numpy(viterbi_rst).cuda())
                # print("Running time of Transformer is {}".format(time.time() - start))

                ##############
                # Run argmax #
                ##############
                pred_list_argmax.append(new_logits.squeeze())

                ############
                # Run mode #
                ############
                count = Counter(sample_list)
                max_count = count.most_common(1)
                mode_rst.append(
                    torch.from_numpy(
                        np.array([int(x) for x in max_count[0][0].split("_")])
                    )
                )

                ####################################
                # Run NLL evalutations starts here #
                ####################################
                bz = all_ref.shape[0]
                gt_sample = np.repeat(gt.cpu().numpy(), bz, axis=0)
                criter = (
                    (gt_sample[:, [0, -1]] == all_ref[:, [0, -1]])
                    .all(axis=1)
                    .nonzero()[0]
                )
                dist_samples = all_ref[criter]
                ref_onehot = torch.FloatTensor(args.pred_horz, act_size).cuda()
                ref_onehot.zero_()

                ######################################################################
                # dist_samples represents the samples in the test-set:               #
                #    1). Share the same start and end-goal semantic;                 #
                #                                                                    #
                # If can not find any dist_samples (aka dist_samples.shape[0] == 0): #
                #    1). Skip the nll evaluation (see below code)                    #
                ######################################################################
                if dist_samples.shape[0] != 0:
                    for vec in dist_samples:
                        vec = torch.from_numpy(vec).cuda()
                        ref_onehot_tmp = torch.FloatTensor(
                            args.pred_horz, act_size
                        ).cuda()
                        ref_onehot_tmp.zero_()
                        ref_onehot_tmp.scatter_(
                            1, vec.view(args.pred_horz, -1), 1)
                        ref_onehot += ref_onehot_tmp

                    ref_dist = ref_onehot

                    """Calculate the nll w.r.t. ref_dist """
                    nll_tmp = []
                    for itm in sample_listing:
                        ###########################################
                        # Convert indivisual sample into onehot() #
                        ###########################################
                        itm_onehot = torch.FloatTensor(
                            args.pred_horz, act_size).cuda()
                        itm_onehot.zero_()
                        itm_onehot.scatter_(
                            1, itm.cuda().view(args.pred_horz, -1), 1)

                        #####################################################
                        # Convert reference distriutions into log_softmax() #
                        #####################################################
                        softmax_logits = torch.nn.functional.softmax(
                            ref_dist, 1
                        ).squeeze()

                        # Truncate extremely small number for numberic steability
                        truncate_val = (torch.ones(
                            1) / 1000000.0).float().cuda()
                        # Softmax the probabilities
                        softmax_logits = torch.where(
                            softmax_logits < truncate_val, truncate_val, softmax_logits
                        )
                        # Log the values
                        log_softmax_logits = torch.log(softmax_logits)

                        ##############################################################################
                        # Reason for this "softmax + log + nll_loss" can be find in Usage example:   #
                        # https://pytorch.org/docs/1.9.0/generated/torch.nn.functional.nll_loss.html #
                        ##############################################################################
                        nll_tmp.append(
                            F.nll_loss(
                                log_softmax_logits.cpu(),
                                itm.cpu().squeeze(),
                            )
                        )
                    nll_rst.append(sum(nll_tmp) / len(nll_tmp))

                ###########################################
                # Evaluate on Mode-Coverage Prec & Recall #
                ###########################################
                ratio_list = []
                for sample in sample_listing:
                    ratio_list.append(
                        (sample.squeeze().cpu().numpy()
                         == dist_samples).all(1).any()
                    )
                ratio = sum(ratio_list) / num_sampling
                mc_prec.append(ratio)

                # all_samples = torch.stack(
                #     sample_listing).squeeze().cpu().numpy()
                all_samples = sample_listing.cpu().numpy()

                num_expert = dist_samples.shape[0]
                list_expert = np.array_split(dist_samples, num_expert)
                tmp_recall = []
                for item in list_expert:
                    tmp_recall.append((item == all_samples).all(1).any())
                mc_recall.append(sum(tmp_recall) / len(tmp_recall))

                ####################################
                #   Calculate the KL-Div  Metric   #
                ####################################
                klv_rst = (
                    custom_KLDiv(
                        sample_softmax_with_temperature(ref_onehot, 0.5),
                        sample_softmax_with_temperature(ref_dist, 0.5),
                    )
                    .cpu()
                    .numpy()
                )
                klv_rst = np.where(np.isnan(klv_rst), 0, klv_rst)
                klv_list.append(klv_rst)

    """ Evaluate using mode results """
    rst_mode = torch.stack(mode_rst)
    rst_mode = rst_mode.view(-1, args.pred_horz)

    """ Evaluate using argmax algorithm """
    rst_argmax = torch.stack(pred_list_argmax)
    rst_argmax = torch.argmax(rst_argmax.view(-1, act_size), 1)
    rst_argmax = rst_argmax.view(-1, args.pred_horz)

    """ Evaluate using viterbi-algorithm """
    rst_viterbi = torch.stack(pred_list)
    rst_viterbi = rst_viterbi.view(-1, args.pred_horz)

    gt = torch.stack(gt_list).squeeze().cpu().numpy().astype("int")
    rst = rst_viterbi.cpu().numpy() + 1
    rst_argmax = rst_argmax.cpu().numpy()
    rst_mode = rst_mode.numpy()

    sr = success_rate(rst, gt, False)
    sr_index = np.argwhere(sr == 1)
    rst_success = rst[sr_index].tolist()

    miou = acc_iou(rst, gt, False)
    macc = mean_category_acc(rst.flatten().tolist(), gt.flatten().tolist())

    avg_mc = sum(mc_prec) / len(mc_prec)
    avg_mc_recall = sum(mc_recall) / len(mc_recall)
    avg_nll = sum(nll_rst) / len(nll_rst)
    avg_ce = sum(klv_list) / len(klv_list)
    avg_entropy = 0.0

    print(
        "For epoch {} using viterbi-algorithm, Best Success Rate {}, meanIOU {}, meanACC {}, Ref-KLDiv {}, MC-Prec {}, MC-Rec {}, Avg.NLL {}".format(
            epoch,
            sr.mean(),
            miou.mean(),
            macc,
            avg_ce,
            avg_mc,
            avg_mc_recall,
            avg_nll,
        )
    )
    sr = success_rate(rst_argmax, gt, False)
    sr_index = np.argwhere(sr == 1)
    rst_success = rst[sr_index].tolist()

    miou = acc_iou(rst_argmax, gt, False)
    macc = mean_category_acc(
        rst_argmax.flatten().tolist(), gt.flatten().tolist())

    print(
        "For epoch {} using argmax, Best Success Rate {}, meanIOU {}, meanACC {} and meanEntropy {}".format(
            epoch,
            sr.mean(),
            miou.mean(),
            macc,
            avg_entropy,
        )
    )

    sr = success_rate(rst_mode, gt, False)
    sr_index = np.argwhere(sr == 1)
    rst_success = rst[sr_index].tolist()

    miou = acc_iou(rst_mode, gt, False)
    macc = mean_category_acc(
        rst_mode.flatten().tolist(), gt.flatten().tolist())

    print(
        "For epoch {} using mode, Best Success Rate {}, meanIOU {}, meanACC {} and meanEntropy {}".format(
            epoch,
            sr.mean(),
            miou.mean(),
            macc,
            avg_entropy,
        )
    )

if __name__ == "__main__":
    train = True
    train = False
    if train:
        for i in range(200):
            train_complete_loss(i, NDR_train=True)
            "Adjust the learning-rate by epoch steps"
            scheduler.step()
            eval(i)
            torch.save(model.state_dict(), os.path.join(
                dir_path, "epoch_{}.pth".format(i)))
    else:
        model_path = (
            os.path.join(
                'checkpoints',
                "CrossTask_best.pth"
            ),
        )
        inference(0, model_path=model_path[0], num_sampling=200)