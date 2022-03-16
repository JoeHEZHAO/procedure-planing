"""
Loader for Multi-Modal dataset
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np
import torch as th
from torch.utils.data import Dataset
import math
import json
import pickle


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


def random_split(task_vids, test_tasks, n_train, seed):
    np.random.seed(seed)
    train_vids = {}
    test_vids = {}

    for task, vids in task_vids.items():
        vids_len = len(vids)
        n_train = int(
            vids_len * 0.7
        )  # default as 70% for training and rest for testing;

        if task in test_tasks and len(vids) > n_train:
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


class CoinTaskDataset(Dataset):
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
        super(CoinTaskDataset, self).__init__()
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

        """Parse the train/val/test vid text file """
        with open(task_vids, "rb") as f:
            files = pickle.load(f)

        with open(constraints_path, "rb") as f:
            coin_annt = json.load(f)

        with open(step_cls_json, "rb") as f:
            step_info = pickle.load(f)

        sequence_len = pred_h

        for vid in files:
            vid_name = vid
            if vid_name in coin_annt["database"]:
                vid_annt = coin_annt["database"][vid_name]
            else:
                self.miss_vid.append(vid_name)
                continue

            """Check if npy file exists... 
            For now, we are missing 54 files, which is ok?
            """
            npy_path = os.path.join(
                self.features_path,
                vid_annt["class"]
                + "_"
                + str(vid_annt["recipe_type"])
                + "_"
                + vid
                + ".npy",
            )
            if not os.path.exists(npy_path):
                self.miss_vid.append(npy_path)
                continue  # Pass on missing vids

            C = []
            W = []
            L = []
            for annt in vid_annt["annotation"]:
                C.append([int(x) for x in annt["segment"]])
                L.append(annt["label"])
                W.append(int(annt["id"]))

            """One more step to generate sequence_len examples """
            if len(W) <= sequence_len:
                continue

            tmp_W = [
                W[i: i + sequence_len] for i in range(len(W) - (sequence_len - 1))
            ]
            tmp_C = [
                C[i: i + sequence_len] for i in range(len(C) - (sequence_len - 1))
            ]
            tmp_L = [
                L[i: i + sequence_len] for i in range(len(L) - (sequence_len - 1))
            ]
            tmp_step_idx = [
                (i, i + sequence_len) for i in range(len(W) - (sequence_len - 1))
            ]

            for w, c, l, ind in zip(tmp_W, tmp_C, tmp_L, tmp_step_idx):
                self.plan_vids.extend(
                    [
                        (
                            vid_annt["class"]
                            + "_"
                            + str(vid_annt["recipe_type"])
                            + "_"
                            + vid,
                            w,
                            c,
                            l,
                            ind,
                        )
                    ]
                )

        print(
            "Original CrossTask dataset had {} dps and SlidingWindow with len {} has {} dps".format(
                len(self.vids), sequence_len, len(self.plan_vids)
            )
        )
        if train:
            print("Due to expired YouTube-video link, missing vid npy files {} for {}".format(
                len(self.miss_vid), 'train'))
        else:
            print("Due to expired YouTube-video link, missing vid npy files {} for {}".format(
                len(self.miss_vid), 'valiation'))

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
        vid, task_cls, C, L, ind = self.plan_vids[idx]

        X = np.load(os.path.join(self.features_path,
                    vid + ".npy"), allow_pickle=True)

        frames = X["frames_features"]
        steps = X["steps_features"]

        if self.mean_vis is not None:
            frames = (X["frames_features"] - self.mean_vis) / \
                np.sqrt(self.var_vis)

        if self.mean_lan is not None:
            steps = (X["steps_features"] - self.mean_lan) / \
                np.sqrt(self.var_lan)

        tmp_X = []
        for idx, (i, _) in enumerate(C):
            frame_idx = min(frames.shape[0] - 2, i)
            tmp_X.append(
                th.stack(
                    [
                        th.tensor(frames[frame_idx - 1]),
                        th.tensor(frames[frame_idx]),
                        th.tensor(frames[frame_idx + 1]),
                    ]
                )
            )

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

        C = th.tensor(C)
        W = th.tensor(task_cls)
        X = th.stack(tmp_X)
        """Shape of X is [4, 3, 512] """
        L = th.tensor(steps[ind[0]: ind[1]])
        """Find the global action indexing """
        return {"vid": vid, "X": X, "C": C, "W": W, "L": L}
