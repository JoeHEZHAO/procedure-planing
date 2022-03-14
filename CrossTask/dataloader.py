"""
Loader for Multi-Modal dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import pdb
import numpy as np
import torch as th
from torch.utils.data import Dataset
import math
import random


def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path, "r") as f:
        idx = f.readline()
        while idx is not "":
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(",")
            next(f)
            idx = f.readline()
    return {"title": titles, "url": urls, "n_steps": n_steps, "steps": steps}


def get_vids(path):
    task_vids = {}
    with open(path, "r") as f:
        for line in f:
            task, vid, url = line.strip().split(",")
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids


def read_plan_assignment(path, task_id, cls_step_json):
    """
    Remove background frames

    Keep start/end frame indices of each action-steps, so that we can index (start -1, start +1) pairs
    """
    # Y = np.zeros([T, K], dtype=np.uint8)
    Y = []
    W = []
    with open(path, "r") as f:
        for line in f:
            # print(line)
            step, start, end = line.strip().split(",")
            start = int(math.floor(float(start)))
            # end = int(math.ceil(float(end)))
            end = int(math.floor(float(end)))
            step = int(step) - 1

            """find the step cls id here"""
            # print(step, task_id)
            step_name = cls_step_json[task_id]["step_name"][str(step)]
            step_cls = cls_step_json[task_id]["steps"][step_name]
            Y.append((start, end))
            W.append(step_cls)
    return Y, W


def read_assignment(T, K, path):
    Y = np.zeros([T, K], dtype=np.uint8)
    with open(path, "r") as f:
        for line in f:
            # print(line)
            step, start, end = line.strip().split(",")
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1
            Y[start:end, step] = 1
    return Y


def random_split_v1(task_vids, test_tasks, n_train, seed):
    np.random.seed(seed)
    train_vids = {}
    test_vids = {}

    for task, vids in task_vids.items():
        vids_len = len(vids)
        n_train = int(
            # vids_len * 0.7
            vids_len
            * 1.0
        )  # default as 70% for training and rest for testing;

        if task in test_tasks and len(vids) >= n_train:
            train_vids[task] = np.random.choice(
                vids, n_train, replace=False).tolist()
            test_vids[task] = [
                vid for vid in vids if vid not in train_vids[task]]
        else:
            train_vids[task] = vids
    return train_vids, test_vids


def random_split(task_vids, test_tasks, n_train, seed):
    np.random.seed(seed)
    train_vids = {}
    test_vids = {}

    for task, vids in task_vids.items():
        vids_len = len(vids)
        n_train = int(
            vids_len * 0.7
        )  # default as 70% for training and rest for testing;

        if task in test_tasks and len(vids) >= n_train:
            train_vids[task] = np.random.choice(
                vids, n_train, replace=False).tolist()
            test_vids[task] = [
                vid for vid in vids if vid not in train_vids[task]]
        else:
            train_vids[task] = vids
    return train_vids, test_vids


def get_A(task_steps, share="words"):
    """Step-to-component matrices."""
    if share == "words":
        # share words
        task_step_comps = {
            task: [step.split(" ") for step in steps]
            for task, steps in task_steps.items()
        }
    elif share == "task_words":
        # share words within same task
        task_step_comps = {
            task: [
                [task + "_" + tok for tok in step.split(" ")] for step in steps]
            for task, steps in task_steps.items()
        }
    elif share == "steps":
        # share whole step descriptions
        task_step_comps = {
            task: [[step] for step in steps] for task, steps in task_steps.items()
        }
    else:
        # no sharing
        task_step_comps = {
            task: [[task + "_" + step] for step in steps]
            for task, steps in task_steps.items()
        }
    vocab = []
    for task, steps in task_step_comps.items():
        for step in steps:
            vocab.extend(step)
    vocab = {comp: m for m, comp in enumerate(set(vocab))}
    M = len(vocab)
    A = {}
    for task, steps in task_step_comps.items():
        K = len(steps)
        a = th.zeros(M, K)
        for k, step in enumerate(steps):
            a[[vocab[comp] for comp in step], k] = 1
        a /= a.sum(dim=0)
        A[task] = a
    return A, M


class CrossTaskDataset(Dataset):
    def __init__(
        self,
        task_vids,
        n_steps,
        features_path,
        constraints_path,
        step_cls_json,
        short_clip=True,
        pred_h=3,
        act_json=None,
        train=True,
    ):
        super(CrossTaskDataset, self).__init__()
        self.vids = []
        self.n_steps = n_steps
        self.features_path = features_path
        self.constraints_path = constraints_path
        self.step_cls_json = step_cls_json
        self.plan_vids = []
        self.miss_vid = []
        self.val_len_vid = []
        self.short_clip = short_clip
        if act_json is not None:
            self.act_json = act_json

        for task, vids in task_vids.items():
            for vid in vids:
                cnst_path = os.path.join(
                    self.constraints_path, task + "_" + vid + ".csv"
                )
                multimodal_data_path = os.path.join(
                    self.constraints_path.split("annotations")[0].split(
                        "crosstask_release"
                    )[0]
                    + "processed_data",
                    task + "_" + vid + ".npy",
                )

                ori_data_path = os.path.join(
                    self.constraints_path.split("annotations")[0].split(
                        "crosstask_release"
                    )[0]
                    + "crosstask_features",
                    vid + ".npy",
                )

                """ Check both the annotations and the npy folders """
                if (
                    os.path.exists(cnst_path)
                    and os.path.exists(multimodal_data_path)
                    and os.path.exists(ori_data_path)
                ):
                    self.vids.extend([(task, vid)])
                else:
                    self.miss_vid.append(vid)

        """One more step to generate sequence_len examples """
        sequence_len = pred_h  # total sequence length at least is pred_h + 2
        shorter = 0.0
        for idx in range(len(self.vids)):
            task, vid = self.vids[idx]
            cnst_path = os.path.join(
                self.constraints_path, task + "_" + vid + ".csv")
            C, W = read_plan_assignment(cnst_path, task, self.step_cls_json)

            if self.short_clip:

                if len(W) <= sequence_len:
                    shorter += 1
                    continue

                self.val_len_vid.append(vid)

                """Opt1: How to pre-process the repeated actions? """
                # ordered_act = list(self.step_cls_json[task]['steps'].values())
                # tmp_W = []
                # tmp_C = []
                # tmp_step_idx = []
                # for i in range(len(W)):
                #     sub_W = [W[i]] ## init the sub_W
                #     sub_C = [C[i]]
                #     sub_idx = [i]
                #     j = i + 1
                #     while len(sub_W) < sequence_len and j < len(W):
                # """This fashion is to ensure the order, which however is not always true """
                # if ordered_act.index(W[j]) > ordered_act.index(sub_W[-1]):
                #     sub_W.append(W[j])
                #     sub_C.append(C[j])
                #     sub_idx.append(j)

                # """This fashion is to ensure none repeated action """
                # if W[j] not in sub_W:
                #     sub_W.append(W[j])
                #     sub_C.append(C[j])
                #     sub_idx.append(j)

                # if ordered_act.index(W[j]) > ordered_act.index(sub_W[-1]) and W[j] not in sub_W:
                #     sub_W.append(W[j])
                #     sub_C.append(C[j])
                #     sub_idx.append(j)

                #                         j += 1
                #
                #                     if len(sub_W) == sequence_len:
                #                         tmp_W.append(sub_W)
                #                         tmp_C.append(sub_C)
                #                         tmp_step_idx.append(sub_idx)

                """Opt2: sliding window to shorter sequences """
                tmp_W = [
                    W[i: i + sequence_len] for i in range(len(W) - (sequence_len - 1))
                ]
                tmp_C = [
                    C[i: i + sequence_len] for i in range(len(C) - (sequence_len - 1))
                ]
                tmp_step_idx = [
                    (i, i + sequence_len) for i in range(len(W) - (sequence_len - 1))
                ]
            else:
                """Use the entire sequence """
                tmp_W = [W]
                tmp_C = [C]
                tmp_step_idx = list(range(len(W)))

            for w, c, ind in zip(tmp_W, tmp_C, tmp_step_idx):
                self.plan_vids.extend([(task, vid, w, c, ind)])

                """Reverse the sequence ==> bi-directional augmentation """
                """Turn out to be not useful """
                # if train:
                # self.plan_vids.extend([(task, vid, w[::-1], c[::-1], ind[::-1])])

        print(
            "Original CrossTask dataset had {} dps and SlidingWindow with len {} has {} dps, shorter sequence {}, missing {} videos".format(
                len(self.vids),
                sequence_len,
                len(self.plan_vids),
                shorter,
                len(self.miss_vid),
            )
        )

    def random_eval_split(self, ratio=0.8, seed=99999999):
        """Use random seed """
        random.seed(seed)
        data_len = len(self.plan_vids)
        n_train = int(
            data_len * ratio
        )  # default as 70% for training and rest for testing;

        self.train_plan_vids = random.sample(self.plan_vids, n_train)

        self.eval_plan_vids = [
            vid for vid in self.plan_vids if vid not in self.train_plan_vids
        ]
        self.plan_vids = self.train_plan_vids

    def random_split(self, ratio=0.7, seed=99999999):
        """Use random seed """
        random.seed(seed)
        data_len = len(self.plan_vids)
        n_train = int(
            data_len * ratio
        )  # default as 70% for training and rest for testing;

        self.train_plan_vids = random.sample(self.plan_vids, n_train)

        self.test_plan_vids = [
            vid for vid in self.plan_vids if vid not in self.train_plan_vids
        ]
        self.plan_vids = self.train_plan_vids

    def __len__(self):
        return len(self.plan_vids)

    def data_whitening(self):

        data_X = []
        data_L = []
        for (task, vid, W, C, ind) in self.plan_vids:
            X = np.load(
                os.path.join(
                    self.features_path.split("crosstask_features")[0]
                    + "processed_data",
                    task + "_" + vid + ".npy",
                ),
                allow_pickle=True,
            )
            data_X.append(X["frames_features"])
            data_L.append(X["steps_features"])  # do not use this for now

        self.mean_vis = np.mean(np.concatenate(data_X, 0))
        self.mean_lan = np.mean(np.concatenate(data_L, 0))
        self.var_vis = np.var(np.concatenate(data_X, 0))
        self.var_lan = np.var(np.concatenate(data_L, 0))

    def __getitem__(self, idx):
        task, vid, W, C, ind = self.plan_vids[idx]
        task_cls = self.act_json[task]

        """Checking only, does not need for official training """
        # if not os.path.exists(os.path.join(self.features_path,task + '_' + vid+'.npy')):
        #     self.miss_vid.append(task + '_' + vid + '.mp4')
        #     # print("{} not exist!!!!!!!!!!!!".format(os.path.join(self.features_path,task + '_' + vid+'.npy')))
        #     return {'vid': vid, 'task': task, 'X': 0, 'C': C, 'W':W}

        X = np.load(
            os.path.join(
                self.features_path.split("crosstask_features")[
                    0] + "processed_data",
                task + "_" + vid + ".npy",
            ),
            allow_pickle=True,
        )

        if self.mean_lan is not None:
            frames = (X["frames_features"] - self.mean_vis) / \
                np.sqrt(self.var_vis)
            steps = (X["steps_features"] - self.mean_lan) / \
                np.sqrt(self.var_lan)
        else:
            frames = X["frames_features"]
            steps = X["steps_features"]

            # starts = X['steps_starts'] # do not use this for now;

        """Indexing to valid positions and totally ignore the white space """
        # tmp_X = [(th.tensor(frames[i-1]) + th.tensor(frames[i+1]) + th.tensor(frames[i])) / 3.0 if idx == 0 else \
        #           (th.tensor(frames[C[idx-1][1]] -1) + th.tensor(frames[i]) + th.tensor(frames[i+1])) / 3.0 for idx, (i, _) in enumerate(C)]

        """Output the concatenate of three frames [3, feat_dim] shape feature for each time-stamps """

        """Way to generate sequence for len(C) = pred_horz + 1"""
        # tmp_X = [
        # th.stack(
        # [
        # th.tensor(frames[i - 1]),
        # th.tensor(frames[i + 1]),
        # th.tensor(frames[i]),
        # ]
        # )
        # if idx == 0
        # else th.stack(
        # [
        # th.tensor(frames[C[idx - 1][1]] - 1),
        # th.tensor(frames[i]),
        # th.tensor(frames[i + 1]),
        # ]
        # )
        # for idx, (i, _) in enumerate(C)
        # ]

        """Way to generate sequence for len(C) = pred_horz"""
        tmp_X = [
            th.stack(
                [
                    th.tensor(frames[i - 1]),
                    th.tensor(frames[i + 1]),
                    th.tensor(frames[i]),
                ]
            )
            if idx == 0
            else th.stack(
                [
                    th.tensor(frames[C[idx - 1][1]] - 1),
                    # th.tensor(frames[i - 1]),
                    th.tensor(frames[i]),
                    th.tensor(frames[i + 1]),
                ]
            )
            for idx, (i, _) in enumerate(C)
        ]

        """Append the final goal observation """
        last_frame = min(C[-1][-1], frames.shape[0] - 2)
        tmp_X.append(
            th.stack(
                [
                    th.tensor(frames[last_frame - 1]),
                    th.tensor(frames[last_frame]),
                    th.tensor(frames[last_frame + 1]),
                ]
            )
        )
        # if C[-1][1] + 1 < frames.shape[0]:
        # tmp_X.append(
        # th.stack(
        # [
        # th.tensor(frames[C[-1][1] - 1]),
        # th.tensor(frames[C[-1][1]]),
        # th.tensor(frames[C[-1][1] + 1]),
        # ]
        # )
        # )
        # elif C[-1][1] < frames.shape[0]:
        # tmp_X.append(
        # th.stack(
        # [
        # th.tensor(frames[C[-1][1] - 2]),
        # th.tensor(frames[C[-1][1] - 1]),
        # th.tensor(frames[C[-1][1]]),
        # ]
        # )
        # )
        # elif C[-1][1] - 1 < frames.shape[0]:
        # tmp_X.append(
        # th.stack(
        # [
        # th.tensor(frames[C[-1][1] - 3]),
        # th.tensor(frames[C[-1][1] - 2]),
        # th.tensor(frames[C[-1][1] - 1]),
        # ]
        # )
        # )
        # else:
        # tmp_X.append(
        # th.stack(
        # [
        # th.tensor(frames[-3]),
        # th.tensor(frames[-2]),
        # th.tensor(frames[-1]),
        # ]
        # )
        # )

        X_ori = np.load(
            os.path.join(
                self.features_path.split("crosstask_features")[0]
                + "crosstask_features",
                vid + ".npy",
            ),
            allow_pickle=True,
        )

        # if self.mean_lan is not None:
        # frames = (X["frames_features"] - self.mean_vis) / np.sqrt(self.var_vis)
        # steps = (X["steps_features"] - self.mean_lan) / np.sqrt(self.var_lan)
        # else:
        # frames = X["frames_features"]
        # steps = X["steps_features"]
        # frames = (X_ori - self.mean_vis) / np.sqrt(self.var_vis)
        
        frames = X_ori

        # starts = X['steps_starts'] # do not use this for now;

        """Indexing to valid positions and totally ignore the white space """
        # tmp_X = [(th.tensor(frames[i-1]) + th.tensor(frames[i+1]) + th.tensor(frames[i])) / 3.0 if idx == 0 else \
        #           (th.tensor(frames[C[idx-1][1]] -1) + th.tensor(frames[i]) + th.tensor(frames[i+1])) / 3.0 for idx, (i, _) in enumerate(C)]

        """Output the concatenate of three frames [3, feat_dim] shape feature for each time-stamps """

        """Way to generate sequence for len(C) = pred_horz + 1"""
        # tmp_X = [
        # th.stack(
        # [
        # th.tensor(frames[i - 1]),
        # th.tensor(frames[i + 1]),
        # th.tensor(frames[i]),
        # ]
        # )
        # if idx == 0
        # else th.stack(
        # [
        # th.tensor(frames[C[idx - 1][1]] - 1),
        # th.tensor(frames[i]),
        # th.tensor(frames[i + 1]),
        # ]
        # )
        # for idx, (i, _) in enumerate(C)
        # ]

        """Way to generate sequence for len(C) = pred_horz"""
        #"""
        tmp_X_ori = [
            th.stack(
                [
                    th.tensor(frames[i - 1]),
                    th.tensor(frames[i + 1]),
                    th.tensor(frames[i]),
                ]
            )
            if idx == 0
            else th.stack(
                [
                    th.tensor(frames[C[idx - 1][1]] - 1),
                    # th.tensor(frames[i - 1]),
                    th.tensor(frames[i]),
                    th.tensor(frames[i + 1]),
                ]
            )
            for idx, (i, _) in enumerate(C)
        ]
       
        #Append the final goal observation
        #If for MIL-NCE training, the final goal is not necessary?
        last_frame = min(C[-1][-1], frames.shape[0] - 2)
        tmp_X_ori.append(
            th.stack(
                [
                    th.tensor(frames[last_frame - 1]),
                    th.tensor(frames[last_frame]),
                    th.tensor(frames[last_frame + 1]),
                ]
            )
        )
        #"""

        # """Append the final goal observation """
        # if C[-1][1] + 1 < frames.shape[0]:
        # tmp_X_ori.append(
        # th.stack(
        # [
        # th.tensor(frames[C[-1][1] - 1]),
        # th.tensor(frames[C[-1][1]]),
        # th.tensor(frames[C[-1][1] + 1]),
        # ]
        # )
        # )
        # elif C[-1][1] < frames.shape[0]:
        # tmp_X_ori.append(
        # th.stack(
        # [
        # th.tensor(frames[C[-1][1] - 2]),
        # th.tensor(frames[C[-1][1] - 1]),
        # th.tensor(frames[C[-1][1]]),
        # ]
        # )
        # )
        # elif C[-1][1] - 1 < frames.shape[0]:
        # tmp_X_ori.append(
        # th.stack(
        # [
        # th.tensor(frames[C[-1][1] - 3]),
        # th.tensor(frames[C[-1][1] - 2]),
        # th.tensor(frames[C[-1][1] - 1]),
        # ]
        # )
        # )
        # else:
        # tmp_X_ori.append(
        # th.stack(
        # [
        # th.tensor(frames[-3]),
        # th.tensor(frames[-2]),
        # th.tensor(frames[-1]),
        # ]
        # )
        # )

        """To verify the chosed indices are correct """
        tmp_indices = [
            (th.tensor(i - 1), th.tensor(i + 1), th.tensor(i))
            if idx == 0
            else (th.tensor(C[idx - 1][1] - 1), th.tensor(i), th.tensor(i + 1))
            for idx, (i, _) in enumerate(C)
        ]

        """Indexing to valid positions but include the white space """
        # tmp_X = [(th.tensor(frames[i-1]) + th.tensor(frames[i+1]) + th.tensor(frames[i])) / 3.0 for idx, (i, _) in enumerate(C)]
        # tmp_X = [(th.tensor(frames[i+1]) + th.tensor(frames[i])) / 2.0 for (i, _) in C]
        # tmp_X = [(th.tensor(frames[i+1])) / 1.0 for (i, _) in C]

        """Random selection of frames"""
        # tmp_X = [th.tensor(frames[np.random.randint(i, j)]) if i < j else th.tensor(frames[i])  for (i, j) in C]

        steps_array = steps.copy()

        if self.short_clip:
            """This fashion generates 4 indices"""
            # tmp_L = steps_array[ind]

            """This fashion only generates 2, the start and end"""
            if ind[0] > ind[1]:  # If in reverse order
                tmp_L = np.flip(steps_array[ind[1]: ind[0]], 0)
            else:
                tmp_L = steps_array[ind[0]: ind[1]]

        else:
            tmp_L = steps_array

        # tmp_X = [th.tensor(frames[np.random.randint(i, j)]) for (i, j) in C]
        C = th.tensor(C)
        W = th.tensor(W)
        X = th.stack(tmp_X)
        X_ori = th.stack(tmp_X_ori)
        X = th.cat([X, X_ori[:, :, -128:]], -1)
        """Shape of X is [4, 3, 512 + 3200] """

        T = th.tensor([task_cls])
        tmp_L = tmp_L.copy()
        L = th.from_numpy(tmp_L)

        """Find the global action indexing """
        return {"vid": vid, "task": task, "X": X, "C": C, "W": W, "T": T, "L": L}
        # return {"vid": vid, "task": task, "X": X_ori, "C": C, "W": W, "T": T, "L": L}
