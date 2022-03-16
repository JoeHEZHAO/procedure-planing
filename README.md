# P3IV: Probabilistic Procedure Planning from Instructional Videos with Weak Supervision

*[He Zhao](https://joehezhao.github.io/)*<sup>1,2</sup>, 
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
<div align='left'>
  <img src="img/cvpr_pic1.jpg" style="display: inline; border-width: 0px;" width=310px></img>
</div>
<div align='right'>
  <img src="img/cvpr_pic2.jpg" style="display: inline; border-width: 0px;" width=610px></img>
</div>
