# ADL2019
Applied Deep Learning (2019 Spring) @ NTU

This course is lectured by Yun-Nung (Vivian) Chen and has four homeworks. The four homeworks are as follows:
   1. Dialogue Modeling
   2. Contextual Embeddings
   3. Deep Reinforcement Learning
   4. Conditional Generative Adversarial Nets
   
Browse this [course website](https://www.csie.ntu.edu.tw/~miulab/s107-adl/index.html) for more details.

# Table of Contents
<!--ts-->
   1. [Dialogue Modeling](https://github.com/JasonYao81000/ADL2019/tree/master/hw1#adl2019hw1)
      1. [Data Preprocessing](https://github.com/JasonYao81000/ADL2019/tree/master/hw1#1-data-preprocessing)
      2. [Training and Prediction](https://github.com/JasonYao81000/ADL2019/tree/master/hw1#2-training-and-prediction)
      3. [Results (Recall@10)](https://github.com/JasonYao81000/ADL2019/tree/master/hw1#3-results-recall10)
   2. [Sequence Classification with Contextual Embeddings](https://github.com/JasonYao81000/ADL2019/tree/master/hw2#adl2019hw2)
      1. [Part 1. Train an ELMo to beat the simple baseline](https://github.com/JasonYao81000/ADL2019/tree/master/hw2#part-1-train-an-elmo-to-beat-the-simple-baseline)
      2. [Part 2. Beat the strong baseline with nearly no limitation](https://github.com/JasonYao81000/ADL2019/tree/master/hw2#part-2-beat-the-strong-baseline-with-nearly-no-limitation)
   3. [Deep Reinforcement Learning](https://github.com/JasonYao81000/ADL2019/tree/master/hw3#adl2019hw3)
      1. [Policy Gradient](https://github.com/JasonYao81000/ADL2019/tree/master/hw3#1-policy-gradient)
      2. [Deep Q-Learning (DQN)](https://github.com/JasonYao81000/ADL2019/tree/master/hw3#2-deep-q-learning-dqn)
      3. [Actor-Critic](https://github.com/JasonYao81000/ADL2019/tree/master/hw3#3-actor-critic)
   4. [Conditional Generative Adversarial Nets](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#adl2019hw4)
      1. [Cartoon Set](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#1-cartoon-set)
      2. [Evaluation](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#2-evaluation)
      3. [Train Condiction GANs](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#3-train-condiction-gans)
      4. [Training Tips for Improvement](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#4-training-tips-for-improvement)
      5. [Evaluate Condiction GANs](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#5-evaluate-condiction-gans)
      6. [FID Scores](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#6-fid-scores)
      7. [Training Progress](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#7-training-progress)
      8. [Loss and Accuracy](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#8-loss-and-accuracy)
      9. [Human Evaluation Results](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#9-human-evaluation-results)
<!--te-->

# Results of Four Homeworks

## 1. Dialogue Modeling
* [README](https://github.com/JasonYao81000/ADL2019/tree/master/hw1#adl2019hw1)

## 2. Contextual Embeddings
* [README](https://github.com/JasonYao81000/ADL2019/tree/master/hw2#adl2019hw2)

## 3. Deep Reinforcement Learning
* [README](https://github.com/JasonYao81000/ADL2019/tree/master/hw3#adl2019hw3)

### 1. Policy Gradient
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000000.gif" width="40%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000001.gif" width="40%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000008.gif" width="40%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000027.gif" width="40%">

### 2. Deep Q-Learning (DQN)
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000000.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000001.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000008.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000027.gif" width="25%">

### 3. Actor-Critic

| World\Stage |  1  |  2  |  3  |  4  |
| :---------: | :-: | :-: | :-: | :-: |
| 1 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-1-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-1-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-1-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-1-4-v0.gif"> |
| 2 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-2-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-2-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-2-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-2-4-v0.gif"> |
| 3 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-3-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-3-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-3-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-3-4-v0.gif"> |
| 4 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-4-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-4-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-4-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-4-4-v0.gif"> |
| 5 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-5-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-5-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-5-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-5-4-v0.gif"> |
| 6 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-6-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-6-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-6-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-6-4-v0.gif"> |
| 7 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-7-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-7-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-7-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-7-4-v0.gif"> |
| 8 | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-8-1-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-8-2-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-8-3-v0.gif"> | <img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/a2c-all-all/SuperMarioBros-8-4-v0.gif"> |

## 4. Conditional Generative Adversarial Nets
* [README](https://github.com/JasonYao81000/ADL2019/tree/master/hw4#adl2019hw4)

### 1. Training Progress
* Resnet-based ACGAN with BCE loss (resnet_1000)
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_1000/resnet_1000.gif" width="100%">

### 2. Human Evaluation Results
* Resnet-based ACGAN with BCE loss (resnet_1000)
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw4/eval_images/resnet_1000/results2.png" width="100%">
