import sys
import os
import torch
import six
import argparse
import numpy as np
import tensorflow as tf

import lmdb
import msgpack
import pyarrow as pa

try:
    import pickle5 as pickle
except:
    import pickle

from os import path as osp
from tqdm import tqdm
from glob import glob

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))  # add parent dir
from models.encoder import Encoder
from TestDecode import (
    decode,
    sample_and_preprocess,
    decode_penn,
    sample_and_preprocess_penn,
)

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source", type=str, help="root folder of tfrecords")
parser.add_argument(
    "-d", "--dest", type=str, help="root folder for storing lmdb folder/files"
)
parser.add_argument(
    "--dataset",
    type=str,
    default="COIN",
    choices=["COIN", "CrossTask", "YouCook2", "PennAction"],
    help="name of the dataset we are encoding",
)
parser.add_argument(
    "--num_parts", type=int, default=1, help="in how many parts to split encoding"
)
parser.add_argument(
    "--part",
    type=int,
    default=1,
    help="the current part of encoding. The value \in [1, num_parts]",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
net = Encoder()
net.to(device)

PENN_DATASETS = {
    "baseball_pitch": 0,
    "baseball_swing": 1,
    "bench_press": 2,
    "bowl": 3,
    "clean_and_jerk": 4,
    "golf_swing": 5,
    "jumping_jacks": 6,
    "pushup": 7,
    "pullup": 8,
    "situp": 9,
    "squat": 10,
    "tennis_forehand": 11,
    "tennis_serve": 12,
}


def process_tfrec_dict(video_dict, dataset):
    vd = video_dict

    # general video-level info
    name = vd["name"].numpy().decode("utf-8")
    if dataset == "COIN":
        activity_name, activity_id = name.split("_")[:2]
    elif dataset == "CrossTask":
        activity_name = activity_id = name.split("_")[0]
    elif dataset == "YouCook2":
        activity_name = activity_id = name.split("_")[0]
    elif dataset == "PennAction":
        activity_name = "_".join(name.split("_")[:-1])
        activity_id = PENN_DATASETS[activity_name]

    cls_id = int(activity_id)

    if dataset != "PennAction":
        duration = vd["duration"].numpy()
        if dataset in ["COIN", "CrossTask"]:
            start = vd["steps_st"].numpy()
            end = vd["steps_ed"].numpy()
        else:
            start = -np.ones([1])
            end = -np.ones([1])

        # video frames embeddings
        frames = vd["frames"].numpy().astype("float")
        frames = frames / 255
        video_frames = torch.from_numpy(frames).to(torch.float32).to(device)

        with torch.no_grad():
            frames_features = net.embed_full_video(video_frames).cpu().detach().numpy()

            # create embeddings for the text that corresponds to steps. Something like:
            step_texts = [s.decode("utf-8") for s in vd["steps"].numpy()]
            step_features = net.embed_full_subs(step_texts).detach().cpu().numpy()

        # subtitles embeddings
        try:
            subs_texts = [s.decode("utf-8") for s in vd["subtitles"].numpy()]
            subs_features = net.embed_full_subs(subs_texts).cpu().detach().numpy()
            subs_starts = vd["subtitles_st"].numpy()
            subs_ends = vd["subtitles_ed"].numpy()
            num_subs = vd["num_subtitles"].numpy()
        except:
            # no encoded subs found
            subs_features = np.zeros([1, 512])
            subs_starts = -np.ones([1])
            subs_ends = -np.ones([1])
            num_subs = np.zeros([0])

        # steps info
        num_steps = vd["num_steps"].numpy()
        steps_ids = vd["steps_ids"].numpy()
        steps_starts = vd["steps_st"].numpy()
        steps_ends = vd["steps_ed"].numpy()

        sample = {
            "name": name,
            "cls": cls_id,
            "cls_name": activity_name,
            "duration": duration,
            "start": start,
            "end": end,
            "frames_features": frames_features,
            "steps_features": step_features,
            "subs_features": subs_features,
            "subs_starts": subs_starts,
            "subs_ends": subs_ends,
            "num_subs": num_subs,
            "num_steps": num_steps,
            "steps_ids": steps_ids,
            "steps_starts": steps_starts,
            "steps_ends": steps_ends,
        }
    else:
        # video frames embeddings
        frames = vd["frames"].numpy()
        frame_labels = vd["frame_labels"].numpy()
        duration = vd["seq_len"].numpy()
        sample = {
            "name": name,
            "cls": cls_id,
            "cls_name": activity_name,
            "duration": duration,
            "frames": frames,
            "frame_labels": frame_labels,
        }
    return sample


def raw_reader(path):
    with open(path, "rb") as f:
        bin_data = f.read()
    return bin_data


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def encode_numpy(tf_dataset, lmdb_path, dataset, dest, write_frequency=1):

    isdir = osp.isdir(lmdb_path)
    import pickle

    print("Generate numpy files to %s" % lmdb_path)

     # This is important when number of frames is too large. Then first split and then merge.
     # Won't be a problem if has >12G GPU mostly.
    interval = 9999999999999999999999 
    for _, tfrec_sample in enumerate(tf_dataset.take(-1)):
        name = tfrec_sample["name"].numpy().decode("utf-8")
        print( "processing  video clip with size {}".format(tfrec_sample["frames"].shape))

        if os.path.exists(
            os.path.join(dest, name + "npy")
        ):
            print("Nnumpy files {} exist, so pass on it !".format(name + "npy"))
            continue
        elif tfrec_sample["frames"].shape[0] > interval:
            video_split_len = np.ceil(tfrec_sample["frames"].shape[0] / interval)
            frames = tfrec_sample["frames"]
            print("File {} is too big and need to split to {} sub-files".format(name + "npy", video_split_len))

            for i in range(int(video_split_len)):
                if os.path.exists(
                    os.path.join(
                        dest,
                        name + "npy" + str(i),
                    )
                ):
                    continue
                else:
                    tfrec_sample["frames"] = frames[i * interval : (i + 1) * interval]
                    video_sample = process_tfrec_dict(tfrec_sample, dataset=dataset)
                    with open(
                        os.path.join(
                            dest,
                            name + "npy" + str(i),
                        ),
                        "wb+",
                    ) as f:
                        pickle.dump(video_sample, f)
                        print("Flushing database ...")
                        print(
                            "Finishing process dataset {} to numpy files!!!!!".format(
                                name + "npy" + str(i)
                            )
                        )
        else:
            video_sample = process_tfrec_dict(tfrec_sample, dataset=dataset)
            with open(
                os.path.join(
                    dest,
                    name + "npy",
                ),
                "wb+",
            ) as f:
                pickle.dump(video_sample, f)
                print("Flushing database ...")
                print(
                    "Finishing process dataset {} to numpy files!!!!!".format(
                        name + "npy"
                    )
                )

def encode_lmdb(tf_dataset, lmdb_path, dataset, write_frequency=1):
    isdir = osp.isdir(lmdb_path)

    print("Generate LMDB to %s" % lmdb_path)
    db = lmdb.open(
        lmdb_path,
        subdir=isdir,
        # map_size=1099511627776 * 2, readonly=False,
        map_size=1099511627776,
        readonly=False,
        meminit=False,
        map_async=True,
    )

    txn = db.begin(write=True)
    names = []
    for idx, tfrec_sample in enumerate(tf_dataset.take(-1)):
        video_sample = process_tfrec_dict(tfrec_sample, dataset=dataset)
        name = video_sample["name"]
        names.append(name)
        txn.put(u"{}".format(name).encode("ascii"), dumps_pyarrow(video_sample))
        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    # finish iterating through dataset
    txn.commit()
    keys = [u"{}".format(k).encode("ascii") for k in names]
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_pyarrow(keys))
        txn.put(b"__len__", dumps_pyarrow(len(keys)))

    print("Flushing database ...")
    print("Finishing process dataset {} !!!!!".format(lmdb_path))
    db.sync()
    db.close()

def encode_folder(source, dest, dataset_name, val=True):
    tfrecords = glob(osp.join(source, "*.tfrecord"))
    folder_name = osp.basename(source.rstrip("/"))
    lmdb_dest = osp.join(dest, folder_name).replace("tfrecords", "lmdb")
    if not osp.isdir(lmdb_dest):
        os.mkdir(lmdb_dest)

    # assuming lmdb would be stored in the same parent folder as of tfrecords
    for tfrecord in tfrecords:
        tfrecord_name = osp.basename(tfrecord)
        lmdb_path = osp.join(lmdb_dest, tfrecord_name.replace("tfrecord", "lmdb"))

        dataset = tf.data.TFRecordDataset(tfrecord)
        if dataset_name == "PennAction":
            dataset = dataset.map(decode_penn)
            dataset = dataset.map(sample_and_preprocess_penn)
        else:
            dataset = dataset.map(decode)
            dataset = dataset.map(sample_and_preprocess)

        encode_numpy(tf_dataset=dataset, lmdb_path=lmdb_path, dataset=dataset_name, dest=dest)

def count_videos(source):
    tfrecords = glob(osp.join(source, "*.tfrecord"))

    # assuming lmdb would be stored in the same parent folder as of tfrecords
    total_len = 0
    for tfrecord in tfrecords:
        tfrecord_name = osp.basename(tfrecord)
        lmdb_path = osp.join(lmdb_dest, tfrecord_name.replace("tfrecord", "lmdb"))

        dataset = tf.data.TFRecordDataset(tfrecord)
        dataset = dataset.map(decode)
        dataset = dataset.map(sample_and_preprocess)

        for sample in tqdm(dataset.take(-1)):
            total_len += 1
    return total_len

if __name__ == "__main__":
    args = parser.parse_args()
    class_folders = glob(osp.join(args.source, "*/"))

    part_size = int(np.ceil(len(class_folders) / args.num_parts))
    part_start, part_end = part_size * (args.part - 1), part_size * args.part
    folders_to_encode = class_folders[part_start:part_end]
    print("\n Folders to encode:", folders_to_encode, "\n")
    
    encode_folder(args.source, args.dest, args.dataset, val=False)
    print("Done")