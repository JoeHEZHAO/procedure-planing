# P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision

*[He Zhao](https://joehezhao.github.io/)*<sup>2</sup>, 
*[Isma Hadji](http://www.cse.yorku.ca/~hadjisma/)*<sup>1</sup>, 
*[Nikita Dvornik](https://thoth.inrialpes.fr/people/mdvornik/)*<sup>1,3</sup>, 
*[Konstantinos G. Derpanis](https://www.cs.ryerson.ca/kosta/)*<sup>1,2</sup>, 
*[Richard P. Wildes](http://www.cse.yorku.ca/~wildes/)*<sup>1,2</sup>, 
*[Allan D. Jepson](https://www.cs.toronto.edu/~jepson/)*<sup>1</sup>,

<sup>1</sup>Samsung AI Center (SAIC) - Toronto &nbsp;&nbsp;
<sup>2</sup>York University &nbsp;&nbsp;
<sup>3</sup>University of Toronto &nbsp;&nbsp;
* This research was conducted while He was an intern at SAIC-Toronto, funded by Samsung Research.

**Abstract**: In this paper, we study the problem of procedure planning in instructional videos. Here, an agent must produce a plausible sequence of actions that can transform the environment from a given start to a desired goal state. When learning procedure planning from instructional videos, most recent work leverages intermediate visual observations as supervision, which requires expensive annotation efforts to localize precisely all the instructional steps in training videos. In contrast, we remove the need for expensive temporal video annotations and propose a weakly supervised approach by learning from natural language instructions. Our model is based on a transformer equipped with a memory module, which maps the start and goal observations to a sequence of plausible actions. Furthermore, we augment our model with a probabilistic generative module to capture the uncertainty inherent to procedure planning, an aspect largely overlooked by previous work. We evaluate our model on three datasets and show our weakly-supervised approach outperforms previous fully supervised state-of-the-art models on multiple metrics.

## Model
<div align="center">
<img src="img/cvpr_pic1.jpg" width=450px></img>
</div>

## Code
This repository contains PyTorch code for three datasets used in this paper: CrossTask [1], COIN [2] and NIV [3].

## CrossTask
We provide two ways to step-up the dataset for CrossTask [1]. You can
### (i) Use pre-extracted features
```
cd datasets/CrossTask_assets
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_release.zip
wget https://www.di.ens.fr/~dzhukov/crosstask/crosstask_features.zip
wget https://www.eecs.yorku.ca/~hezhao/crosstask_s3d.zip
unzip '*.zip'
```
### Or, extract features from raw video
```
cd raw_data_process
python download_CrossTask_videos.py
python InstVids2TFRecord_CrossTask.py
bash lmdb_encode_CrossTask.sh 1 1
```
### (ii) Train and Evaluation
Set the variable **train** (under **if \_\_name\_\_ == \_\_main\_\_**) to either True/False, to choose between train a network or evaluate a pre-trained model. By default, the code will load the used random datasplit (see datasplit.pth in ./checkpoints) as well as our pre-trained weights (included in ./checkpoints folder).
```
python CrossTask_main.py
```

## COIN
Similarly, to use COIN dataset [2] on our approach, we provide
### (i) Pre-extracted features
```
cd datasets/CrossTask_assets
wget https://www.eecs.yorku.ca/~hezhao/coin_s3d.zip
unzip '*.zip'
```
### or, extract features from raw video
```
cd raw_data_process
python download_COIN_videos.py
python InstVids2TFRecord_COIN.py
bash lmdb_encode_COIN.sh 1 1
```
### (ii) Train and Evaluation
The train/evaluation code for COIN is in the same design before.
```
python COIN_main.py
```

## NIV
For the NIV dataset [3], either use
### (i) Pre-extracted features
```
cd datasets/NIV_assets
wget https://www.eecs.yorku.ca/~hezhao/niv_s3d.zip
unzip '*.zip'
```
### Or, extract features from raw video
Download videos from project page of NIV, unzip it and move all mpg files to **datasets/NIV_assets/videos**
```
cd datasets/NIV_assets/videos
wget https://www.di.ens.fr/willow/research/instructionvideos/data_new.tar.gz
tar -xvzf data_new.tar.gz
find ./data_new -type f -name “*.mpg” | xargs -iF mv F .
```
Then jump to raw_data_process and process raw videos
```
cd raw_data_process
python InstVids2TFRecord_NIV.py
bash lmdb_encode_NIV.sh 1 1
```
### (ii) Train and Evaluation
The train/evaluation code for NIV is in the same design before.
```
python NIV_main.py
```
