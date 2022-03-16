import sys
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import path as osp
import os
import glob
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))    # add parent dir
from dataset_utils import Time2FrameNumber

def decode(serialized_example):
    """Decode serialized SequenceExample."""

    context_features = {
      'name': tf.io.FixedLenFeature([], dtype=tf.string),
      'len': tf.io.FixedLenFeature([], dtype=tf.int64),
      'num_steps': tf.io.FixedLenFeature([], dtype=tf.int64),
      #'num_subtitles': tf.io.FixedLenFeature([], dtype=tf.int64),
      #'start': tf.io.FixedLenFeature([], dtype=tf.float32),
      #'end': tf.io.FixedLenFeature([], dtype=tf.float32),
      'duration': tf.io.FixedLenFeature([], dtype=tf.float32),
      'fps': tf.io.FixedLenFeature([], dtype=tf.float32),
    }
    seq_features = {}

    seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    seq_features['steps'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    seq_features['steps_ids'] = tf.io.FixedLenSequenceFeature([], dtype=tf.int64)
    seq_features['steps_st'] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    seq_features['steps_ed'] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    #seq_features['subtitles'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    #seq_features['subtitles_st'] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
    #seq_features['subtitles_ed'] = tf.io.FixedLenSequenceFeature([], dtype=tf.float32)
  
    # Extract features from serialized data.
    context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=seq_features)
  
    name = tf.cast(context_data['name'], tf.string)
    seq_len = context_data['len']
    num_steps = context_data['num_steps']
    #num_subtitles = context_data['num_subtitles']
    #start = context_data['start']
    #end = context_data['end']
    duration = context_data['duration']
    fps = context_data['fps']

    video = sequence_data.get('video', [])
    steps = sequence_data.get('steps', [])
    steps_ids = sequence_data.get('steps_ids', [])
    steps_st = sequence_data.get('steps_st', [])
    steps_ed = sequence_data.get('steps_ed', [])
    #subtitles = sequence_data.get('subtitles', [])
    #subtitles_st = sequence_data.get('subtitles_st', [])
    #subtitles_ed = sequence_data.get('subtitles_ed', [])

    """
    return seq_len, name,  num_steps, num_subtitles, start, end, duration, fps,\
                     video, steps, steps_ids, steps_st, steps_ed,\
                     subtitles, subtitles_st, subtitles_ed
    """
    return seq_len, name,  num_steps, duration, fps,\
                     video, steps, steps_ids, steps_st, steps_ed

def sample_and_preprocess(seq_len,
                          name,
                          num_steps,
                          #num_subtitles,
                          #start,
                          #end,
                          duration,
                          fps,
                          video,
                          steps,
                          steps_ids,
                          steps_st,
                          steps_ed,
                          #subtitles,
                          #subtitles_st,
                          #subtitles_ed,
                          ):
    """Samples frames and prepares them for training."""
    # Decode the encoded JPEG images
    
    video = tf.map_fn(
      tf.image.decode_jpeg,
      video,
      dtype=tf.uint8)

    #frame_labels = frame_labels
    return {
      'frames': video,
      'steps': steps,
      #'subtitles': subtitles,
      'steps_ids': steps_ids,
      'steps_st': steps_st,
      'steps_ed': steps_ed,
      #'subtitles_st': subtitles_st,
      #'subtitles_ed': subtitles_ed,
      'seq_len': seq_len,
      'name': name,
      'num_steps': num_steps,
      #'num_subtitles': num_subtitles,
      #'start': start,
      #'end': end,
      'duration': duration,
      'fps': fps,
      }

def decode_penn(serialized_example):
    """Decode serialized SequenceExample."""

    context_features = {
    'name': tf.io.FixedLenFeature([], dtype=tf.string),
    'len': tf.io.FixedLenFeature([], dtype=tf.int64),
    }

    seq_features = {}

    seq_features['video'] = tf.io.FixedLenSequenceFeature([], dtype=tf.string)
    seq_features['frame_labels'] = tf.io.FixedLenSequenceFeature(
        [], dtype=tf.int64)
  
    # Extract features from serialized data.
    context_data, sequence_data = tf.io.parse_single_sequence_example(
      serialized=serialized_example,
      context_features=context_features,
      sequence_features=seq_features)
  
    name = tf.cast(context_data['name'], tf.string)
    seq_len = context_data['len']
  
    video = sequence_data.get('video', [])
    frame_labels = sequence_data.get('frame_labels', [])

    return video, frame_labels, seq_len, name

def sample_and_preprocess_penn(video,
                               frame_labels,
                               seq_len,
                               name):
    """Samples frames and prepares them for training."""
    # Decode the encoded JPEG images
    
    video = tf.map_fn(
      tf.image.decode_jpeg,
      video,
      dtype=tf.uint8)

    #frame_labels = frame_labels
    return {
      'frames': video,
      'frame_labels': frame_labels,
      'seq_len': seq_len,
      'name':name
      }

if __name__ == "__main__":
  tfrecords_path = '/Users/isma.hadji/Desktop/tfrecords/'
  in_dir = os.listdir(tfrecords_path)
  count = 0
  for tf_dir in in_dir:
    print(tf_dir)
    tf_dir_path = tfrecords_path + tf_dir
    tfdir_files = glob.glob(tf_dir_path+'/*train.tfrecord')
    for tfrecord_file in tfdir_files:
      dataset = tf.data.TFRecordDataset(tfrecord_file)
      dataset = dataset.map(decode)
      dataset = dataset.map(sample_and_preprocess)
      cc = 0
      for data in dataset.take(1000):
          # this is just a hack to see a specific video I am interested in
          #if count <2:
          #    count += 1
          #    continue
          cc += 1
          #print(data['name'])
          print('total number of videos in %s is %d' % (tf_dir,cc))
      
          
          print(tfrecord_file)
          print(data['seq_len'])
          print(data['name'])
          print(data['num_steps'])
          # print(data['start'])
          # print(data['end'])
          print(data['duration'])
          print(data['fps'])
          print(data['frames'].shape)
          print(data['steps'])
          print(data['steps_ids'])
          print(data['steps_st'])
          print(data['steps_ed'])
          # print(data['num_subtitles'])
          
          print(count)
          input('pause')
          # get start and end frames of each step
          num_steps = data['num_steps'].numpy()
          #num_subtitles = data['num_subtitles'].numpy()
          steps = []
          steps_st = []
          steps_ed = []
          
          for s in range(num_steps):
              steps.append(data['steps'].numpy()[s].decode('utf-8'))
              st = data['steps_st'].numpy()[s]
              ed = data['steps_ed'].numpy()[s]
              steps_st.append(Time2FrameNumber(st, data['fps'].numpy()))
              steps_ed.append(Time2FrameNumber(ed, data['fps'].numpy()))
          
          # subtitles = []
          # subtitles_st = []
          # subtitles_ed = []
          # for s in range(num_subtitles):
          #     subtitles.append(data['subtitles'].numpy()[s].decode('utf-8'))
          #     st = data['subtitles_st'].numpy()[s]
          #     ed = data['subtitles_ed'].numpy()[s]
          #     subtitles_st.append(Time2FrameNumber(st, data['fps'].numpy()))
          #     subtitles_ed.append(Time2FrameNumber(ed, data['fps'].numpy()))
              
          print(steps_st, steps_ed)
          input('pause')
          # visualize video with annotated steps
          video = data['frames'].numpy()
          for frame in range(0,video.shape[0],5):
              plt.imshow(video[frame])
              plt.title('frame %d | NO KEY-STEP' % (frame+1))
              # if you want to see steps
              
              for s in range(num_steps):
                  if (frame >=steps_st[s]) and (frame <= steps_ed[s]):
                      plt.title('frame %d \n STEP %s: %s' % (frame+1, s+1,steps[s]))
              """
              # if you want to see subtitles
              for s in range(num_subtitles):
                  if (frame >=subtitles_st[s]) and (frame <= subtitles_ed[s]):
                      plt.title('frame %d \n STEP %s: %s' % (frame+1, s+1,subtitles[s]))
              """
              plt.xticks([]), plt.yticks([])
              plt.pause(0.01)
              plt.clf()
          input('pause')
          
      count = count + cc
      print('total number of videos is %d' % count)
