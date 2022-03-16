#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 13:23:02 2020

@author: isma.hadji
code to convert instructional videos (with corresponding labels) to tfrecords
"""
import json
import os, sys
import os.path as osp
import glob
import dataset_utils as utils
import tensorflow as tf
from absl import logging
from absl import flags
from absl import app
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from paths import COIN_PATH

# flags.DEFINE_string('mode', 'train', 'define which category of videos to encode train vs. test')
flags.DEFINE_string('mode', 'val', 'define which category of videos to encode train vs. test')
flags.DEFINE_boolean('delete', False, 'flag to delete videos after they are encoded')

FLAGS = flags.FLAGS

feature = tf.train.Feature
bytes_feature = lambda v: feature(bytes_list=tf.train.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=tf.train.Int64List(value=v))
float_feature = lambda v: feature(float_list=tf.train.FloatList(value=v))

def main(_):
    #%% Initialize dataset in & out directories
    input_path = os.path.join(COIN_PATH, 'videos')
    input_path_labels = os.path.join(COIN_PATH, 'COIN.json')
    
    class_list = sorted(os.listdir(input_path))
    # decode json file to obtain labels
    data = json.load(open(input_path_labels, 'r'))['database']
            
    input_dir = input_path + '/'
    # get all examples per class
    in_dir = glob.glob(input_dir+'*.mp4') #os.listdir(input_dir)
    # create tfrecord save destination
    record_name = os.path.join(COIN_PATH, 'COIN_tfrecords.tfrecord')
    print(record_name)
    
    #input('pause')
    writer = tf.io.TFRecordWriter(record_name)
    #%% create tfrecord per class
    cc = 0
    for filename in in_dir:
        video_filename = os.path.basename(filename)
        subtitle_filename = video_filename.replace(".mp4","") + '.en.vtt'

        if (video_filename == '.') or (video_filename == '..'):
            continue
        #elif cc == 3:
        #    break
        else:
            #try:
            cc += 1
            # decode video
            video = input_path + '/' + video_filename
            logging.info('video: %s' %video)
            # process video
            frames_list = utils.video_to_frames_native(video, 
                                                        # fps=10, 
                                                        fps=16, 
                                                        size=224, 
                                                        center_crop=True, 
                                                        crop_only=False)
            ori_fps = 16
            #frames_list, _, ori_fps = utils.video_to_frames(video,
            #                                          fps=10, 
            #                                          resize=True, 
            #                                          crop =True, 
            #                                          max_size = 480, 
            #                                          width=224, 
            #                                          height=224, 
            #                                          rotate=False, 
            #                                          cropFirst=True)
            print(frames_list.shape[0])
            #input('check')
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
                img = frames_list[frame,:,:,:]
                frame_count += 1
                # convert img to bytes for tfrecord
                frames_bytes.append(utils.image_to_bytes(img))
            print(len(frames_list), frame_count)
            #input('pause')
            # get video metadata/labels
            video_name = video_filename.replace(".mp4","")[-11:]
            info = data[video_name]
            start = info['start']
            end = info['end']
            duration = info['duration']
            print(start,end,duration)
            #input('pause')
            # get video corresponding step annotations
            annotation = info['annotation']
            num_steps = len(annotation)
            task_labels = []
            task_ids = []
            segments_st = []
            segments_ed = []
            print(video_filename, num_steps)
            #input('check num steps')
            for i in range(num_steps):
                anno = annotation[i]
                task_ids.append(int64_feature([int(anno['id'])]))
                segments_st.append(float_feature([anno['segment'][0]]))
                segments_ed.append(float_feature([anno['segment'][1]]))
                task_labels.append(bytes_feature([str.encode(anno['label'])]))
                print(anno['id'])
                print(anno['segment'][0], anno['segment'][1])
                print(str.encode(anno['label']))
                #input('pause')
            
            # process subtitles
            subs_path = input_path + '/' + subtitle_filename
            if os.path.exists(subs_path):
                caption_texts, caption_starts, caption_ends = utils.process_subs(subs_path)
            else:
                caption_texts = [str.encode('no captions')]
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
            seq_feats['video'] = tf.train.FeatureList(feature=frames_bytes)
            seq_feats['steps'] = tf.train.FeatureList(feature=task_labels)
            seq_feats['steps_ids'] = tf.train.FeatureList(feature=task_ids)
            seq_feats['steps_st'] = tf.train.FeatureList(feature=segments_st)
            seq_feats['steps_ed'] = tf.train.FeatureList(feature=segments_ed)
            seq_feats['subtitles'] = tf.train.FeatureList(feature=subtitles)
            seq_feats['subtitles_st'] = tf.train.FeatureList(feature=subtitles_st)
            seq_feats['subtitles_ed'] = tf.train.FeatureList(feature=subtitles_ed)
            # Create FeatureLists.
            feature_lists = tf.train.FeatureLists(feature_list=seq_feats)
            # Add context or video-level features
            seq_len = frame_count
            text_name = video_filename.split('_')[0]
            cat = video_filename.split('_')[1]
            name = str.encode(text_name + '_' + cat + '_' + video_name)
            context_features_dict = {'name': bytes_feature([name]),
                                        'len': int64_feature([seq_len]),
                                        'num_steps': int64_feature([num_steps]),
                                        'start': float_feature([start]),
                                        'end': float_feature([end]),
                                        'duration': float_feature([duration]),
                                        'fps': float_feature([ori_fps]),
                                        'num_subtitles': int64_feature([num_subtitles])}
            context_features = tf.train.Features(feature=context_features_dict)
            # Create SequenceExample.
            ex = tf.train.SequenceExample(context=context_features,
                                    feature_lists=feature_lists)
            writer.write(ex.SerializeToString())
            #except:
            #    continue
            
    writer.close()

if __name__ == '__main__':
    app.run(main)
