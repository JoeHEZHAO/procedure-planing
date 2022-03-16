#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Feb 16 13:23:02 2021

@author: isma.hadji
code to convert instructional videos (with corresponding labels) to tfrecords
"""
import os,sys
import os.path as osp
import glob
import dataset_utils as utils
import tensorflow as tf
from absl import logging
from absl import flags
from absl import app
from dataset_utils import read_task_files, read_anno_files
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import CROSSTASK_PATH

flags.DEFINE_string(
    "mode", "train", "define which category of videos to encode train vs. test"
)
# flags.DEFINE_string('mode', 'val', 'define which category of videos to encode train vs. test')
flags.DEFINE_boolean("delete", False, "flag to delete videos after they are encoded")

FLAGS = flags.FLAGS

feature = tf.train.Feature
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))

def main(_):
    #%% Initialize dataset in & out directories
    input_path = os.path.join(CROSSTASK_PATH, "videos/")
    input_path_labels =  os.path.join(CROSSTASK_PATH, "crosstask_release/annotations/")
    task_primary = os.path.join(CROSSTASK_PATH, "crosstask_release/tasks_primary.txt")
    task_related = os.path.join(CROSSTASK_PATH, "crosstask_release/tasks_related.txt")

    # read through annotation folder
    annt_folder = os.path.join(CROSSTASK_PATH, "crosstask_release/annotations")

    annt_files = glob.glob(os.path.join(annt_folder, "*.csv"))
    split_annt_files = [x.split("/")[-1] for x in annt_files]
    split_annt_files_ids = [x[:-16] for x in split_annt_files]
    split_annt_files_names = [x[-15:] for x in split_annt_files]

    in_dir = glob.glob(input_path + "*.mp4")  # all mp4 files to be processed
    exit_vids = glob.glob(
        os.path.join(CROSSTASK_PATH, "processed_data/" + "*.npy")
    )

    new_class_list = []
    already_exist = []
    tobe_proc_vid_list = []
    tobe_proc_id_list = []
    for video in in_dir:
        vid_name = video[-15:].replace("mp4", "csv")
        if vid_name in split_annt_files_names:
            index = split_annt_files_names.index(vid_name)
            task_tmp_id = split_annt_files_ids[index]
            new_class_list.append(task_tmp_id)
            npy_file = os.path.join(
                os.path.join(CROSSTASK_PATH, "processed_data"),
                str(task_tmp_id) + "_" + vid_name.replace("csv", "npy"),
            )

            "Comment below 'if' out when reproducing the whole set"
            if npy_file in exit_vids:
                already_exist.append(npy_file)
            else:
                tobe_proc_vid_list.append(video)
                tobe_proc_id_list.append(task_tmp_id)

    taskp = read_task_files(task_primary)
    taskr = read_task_files(task_related)
    data = {**taskp, **taskr}
    # data = {**taskp}

    "Specify Location to output the tfrecord file"
    tfrecord_path = os.path.join(CROSSTASK_PATH, "crosstask.tfrecord")

    writer = tf.io.TFRecordWriter(tfrecord_path)

    # for filename in in_dir:
    for filename, cat in zip(tobe_proc_vid_list, tobe_proc_id_list):
        video_filename = os.path.basename(filename)
        labels_filename = (
            # cat + "_" + video_filename[:-3] + "csv"
            cat
            + "_"
            + video_filename[:-3]
            + "csv"
        )  # cat should be the task_ids;
        subtitle_filename = video_filename.replace(".mp4", "") + ".en.vtt"
        if (video_filename == ".") or (video_filename == ".."):
            continue
        # elif cc == 3:
        #    break
        else:
            try:
                # cc += 1
                # decode video
                video = os.path.join(input_path, video_filename)
                logging.info("video: %s" % video)

                # process video
                frames_list = utils.video_to_frames_native(
                    video, fps=16, size=224, center_crop=True, crop_only=False
                )
                ori_fps = 16  # 10
                # frames_list, _, ori_fps = utils.video_to_frames(video,
                #                                          fps=10,
                #                                          resize=True,
                #                                          crop =True,
                #                                          max_size = 480,
                #                                          width=224,
                #                                          height=224,
                #                                          rotate=False,
                #                                          cropFirst=True)
                # print(len(frames_list))
                print(frames_list.shape)
                # input('check')
                # to delete videos after they are encoded (to save storage space)
                if FLAGS.delete:
                    if os.path.exists(video):
                        os.remove(video)

                # reset frame counter and seq_feats for each new video
                frame_count = 0
                frames_bytes = []
                # get frames data
                for frame in range(frames_list.shape[0]):
                    # save all video frames as bytes
                    img = frames_list[frame, :, :, :]
                    frame_count += 1
                    # convert img to bytes for tfrecord
                    frames_bytes.append(utils.image_to_bytes(img))

                # get video metadata/labels
                video_labels_path = input_path_labels + labels_filename
                # no global start and end times for cross task
                # but keeping labels for compatibility with COIN format
                start = -1
                end = -1
                # using this variable to store the number of steps needed to describe the task
                # i.e. number of steps as described in wikihow
                duration = float(data[str(cat)]["num_steps"])
                # get steps
                steps = data[str(cat)]["annotations"]
                # get video corresponding step annotations from csv files
                task_labels = []
                task_ids = []
                segments_st = []
                segments_ed = []
                # if this is an annotated video encode annotations
                if os.path.exists(video_labels_path):
                    annotation = read_anno_files(video_labels_path)
                    num_steps = len(annotation)
                    print(video_labels_path, num_steps)
                    for i in range(num_steps):
                        task_id = int(annotation[i][0])
                        task_ids.append(int64_feature([task_id]))
                        segments_st.append(float_feature([float(annotation[i][1])]))
                        segments_ed.append(float_feature([float(annotation[i][2])]))
                        step = steps[task_id - 1]
                        task_labels.append(bytes_feature([str.encode(step)]))
                        print(task_id)
                        print(segments_st[i], segments_ed[i])
                        print(str.encode(step))
                        # input('pause')
                else:
                    # if video is not annotated we can still use the steps info from wikihow
                    num_steps = (
                        -1
                    )  # to indicate that this is not labeled with step ordering
                    for i in range(int(duration)):
                        task_id = i + 1
                        task_ids.append(int64_feature([task_id]))
                        segments_st.append(float_feature([float(-1)]))
                        segments_ed.append(float_feature([float(-1)]))
                        step = steps[i]
                        task_labels.append(bytes_feature([str.encode(step)]))

                # process subtitles
                subs_path = input_path + cat + "/" + subtitle_filename
                if os.path.exists(subs_path):
                    (
                        caption_texts,
                        caption_starts,
                        caption_ends,
                    ) = utils.process_subs(subs_path)
                else:
                    caption_texts = [str.encode("no captions")]
                    caption_starts = [-1]
                    caption_ends = [-1]
                # encode subtitles
                num_subtitles = len(caption_texts)
                print(num_subtitles)
                subtitles = []
                subtitles_st = []
                subtitles_ed = []
                for i in range(num_subtitles):
                    subtitles_st.append(float_feature([caption_starts[i]]))
                    subtitles_ed.append(float_feature([caption_ends[i]]))
                    subtitles.append(bytes_feature([caption_texts[i]]))
                # add this sequence information into tfrecord
                seq_feats = {}
                seq_feats["video"] = tf.train.FeatureList(feature=frames_bytes)
                seq_feats["steps"] = tf.train.FeatureList(feature=task_labels)
                seq_feats["steps_ids"] = tf.train.FeatureList(feature=task_ids)
                seq_feats["steps_st"] = tf.train.FeatureList(feature=segments_st)
                seq_feats["steps_ed"] = tf.train.FeatureList(feature=segments_ed)
                seq_feats["subtitles"] = tf.train.FeatureList(feature=subtitles)
                seq_feats["subtitles_st"] = tf.train.FeatureList(feature=subtitles_st)
                seq_feats["subtitles_ed"] = tf.train.FeatureList(feature=subtitles_ed)
                # Create FeatureLists.
                feature_lists = tf.train.FeatureLists(feature_list=seq_feats)
                # Add context or video-level features
                seq_len = frame_count
                # print(seq_len)
                name = str.encode(video_filename[:-3])
                context_features_dict = {
                    "name": bytes_feature([name]),
                    "len": int64_feature([seq_len]),
                    "num_steps": int64_feature([num_steps]),
                    "start": float_feature([start]),
                    "end": float_feature([end]),
                    "duration": float_feature([duration]),
                    "fps": float_feature([ori_fps]),
                    "num_subtitles": int64_feature([num_subtitles]),
                }
                context_features = tf.train.Features(feature=context_features_dict)
                # Create SequenceExample.
                ex = tf.train.SequenceExample(
                    context=context_features, feature_lists=feature_lists
                )
                writer.write(ex.SerializeToString())
            except:
                continue

    writer.close()

if __name__ == "__main__":
    app.run(main)
