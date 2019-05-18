# ADL2019/hw3

## 0. Requirements
* Install VC++ via Microsoft Visual Studio on the Windows.
```shell
$ conda install swig
$ conda install pytorch==1.0.1 torchvision cudatoolkit=10.0 -c pytorch
$ pip install opencv-python
$ pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py
$ pip install gym gym[box2d] gym[atari]
$ pip install gym-super-mario-bros cloudpickle
```

## 1. Policy Gradient

### 1.1. Introduction
* Game Playing: `LunarLander`
* Game Environment: `LunarLander-v2`
* Implement an agent to play Atari games using Deep Reinforcement Learning.
* In this homework, you are required to implement Policy Gradient.
* Improvements to Policy Gradient:
    * [ ] Variance Reduction
    * [ ] Advanced Advantage Estimation
    * [ ] Off-policy learning by Importance Sampling
    * [ ] Natural Policy Gradient
    * [ ] Trust Region Policy Optimization
    * [x] **Proximal Policy Optimization**
* Training Hint
    * [x] **Reward normalization (More stable)**
    * [ ] Action space reduction

### 1.2. Baseline
* Getting averaging reward in 30 episodes over **0** in LunarLander
* Improvements to Policy Gradient are allowed, not including Actor-Critic series.

### 1.3. Training Policy Gradient on `LunarLander`
1. Training Policy Gradient:
    `$ python main.py --train_pg`
2. Training Policy Gradient with PPO:
    `$ python main.py --train_pg --ppo`

### 1.4. Testing Policy Gradient on `LunarLander`
1. Testing Policy Gradient:
    `$ python main.py --test_pg --video_dir ./results/pg`
2. Testing Policy Gradient with PPO:
    `$ python main.py --test_pg --ppo --video_dir ./results/pg-ppo`

### 1.5 Testing Videos for DQN on `Assault`
1. Policy Gradient

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg/openaigym.video.0.21396.video000000.gif" width="40%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg/openaigym.video.0.21396.video000001.gif" width="40%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg/openaigym.video.0.21396.video000008.gif" width="40%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg/openaigym.video.0.21396.video000027.gif" width="40%">

2. Policy Gradient with PPO

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000000.gif" width="40%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000001.gif" width="40%">

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000008.gif" width="40%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/pg-ppo/openaigym.video.0.13592.video000027.gif" width="40%">

### 1.6. Rewards in 30 Episodes
1. Policy Gradient: `Run 30 episodes, Mean: 159.41205516866356`
2. Policy Gradient with PPO: `Run 30 episodes, Mean: 218.51080037730148`

### 1.7. Learning Curve
* Learning Curve of Original Policy Gradient
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_pg.png" width="100%">

* Learning Curve of Policy Gradient with Proximal Policy Optimization (PPO)
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_pg-ppo.png" width="100%">

* Comparison of Original PG and PG with PPO
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_pgs.png" width="100%">

## 2. Deep Q-Learning (DQN)

### 2.1. Introduction
* Game Playing: `Assault`
* Game Environment: `AssaultNoFrameskip-v0`
* Implement an agent to play Atari games using Deep Reinforcement Learning.
* In this homework, you are required to implement Deep Q-Learning (DQN).
* Improvements to DQN: 
  * [x] **Double Q-Learning**
  * [x] **Dueling Network**
  * [ ] Prioritized Replay Memory
  * [ ] Multi-Step Learning
  * [ ] Noisy DQN
  * [ ] Distributional DQN
* Training Hint
  * The action should act ε-greedily
    * [x] Random action with probability ε
    * [ ] Also in testing
  * Linearly decline ε from 0.9 to some small value, say 0.05
    * [x] Decline per step
    * [x] Randomness is for exploration, agent is weak at start
  * Hyperparameters
    * [x] Replay Memory Size: 10000
    * [x] Perform Update Current Network Step: 4
    * [x] Perform Update Target Network Step: 1000
    * [x] Learning Rate: 1e-4
    * [x] Batch Size: 32

### 2.2. Baseline
* Getting averaging reward in 100 episodes over **100** in Assault
* Improvements to DQN are allowed,  not including Actor-Critic series.

### 2.3. Training DQN on `Assault`
1. Training DQN:
    `$ python main.py --train_dqn`
2. Training Dual DQN:
    `$ python main.py --train_dqn --duel_dqn`
2. Training Double DQN:
    `$ python main.py --train_dqn --double_dqn`
2. Training Double Dual DQN:
    `$ python main.py --train_dqn --double_dqn --duel_dqn`

### 2.4. Testing DQN on `Assault`
1. Testing DQN:
    `$ python main.py --test_dqn --video_dir ./results/dqn`
2. Testing Dual DQN:
    `$ python main.py --test_dqn --duel_dqn --video_dir ./results/duel_dqn`
2. Testing Double DQN:
    `$ python main.py --test_dqn --double_dqn --video_dir ./results/double_dqn`
2. Testing Double Dual DQN:
    `$ python main.py --test_dqn --double_dqn --duel_dqn --video_dir ./results/double_duel_dqn`

### 2.5 Testing Videos for DQN on `Assault`
1. DQN

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000000.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000001.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000008.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/dqn/openaigym.video.0.19364.video000027.gif" width="25%">

2. Dual DQN

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/duel_dqn/openaigym.video.0.2572.video000000.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/duel_dqn/openaigym.video.0.2572.video000001.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/duel_dqn/openaigym.video.0.2572.video000008.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/duel_dqn/openaigym.video.0.2572.video000027.gif" width="25%">

3. Double DQN

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_dqn/openaigym.video.0.14420.video000000.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_dqn/openaigym.video.0.14420.video000001.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_dqn/openaigym.video.0.14420.video000008.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_dqn/openaigym.video.0.14420.video000027.gif" width="25%">

4. Double Dual DQN

<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_duel_dqn/openaigym.video.0.18952.video000000.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_duel_dqn/openaigym.video.0.18952.video000001.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_duel_dqn/openaigym.video.0.18952.video000008.gif" width="25%"><img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/double_duel_dqn/openaigym.video.0.18952.video000027.gif" width="25%">

### 2.6. Rewards in 100 Episodes
1. DQN: `Run 100 episodes, Mean: 204.71`
2. Dual DQN: `Run 100 episodes, Mean: 193.49`
3. Double DQN: `Run 100 episodes, Mean: 188.83`
4. Double Dual DQN: `Run 100 episodes, Mean: 174.99`

### 2.7. Learning Curve
* Learning Curve of DQN
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_dqn.png" width="100%">

* Learning Curve of Dual DQN
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_duel.png" width="100%">

* Learning Curve of Double DQN
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_double.png" width="100%">

* Learning Curve of Double Dual DQN
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_double_duel.png" width="100%">

* Comparison of DQN, Dual DQN, Double DQN and Double Dual DQN
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_dqns.png" width="100%">

## 3. Actor-Critic

### 3.1. Introduction
* Game Playing: `SuperMarioBros`
* Game Environment: `SuperMarioBros-v0`
* Implement an agent to play Super Mario Bros using Actor-Critic.

### 3.2. Baseline
* Simple Baseline: Getting averaging reward in 10 episodes over **1500** in SuperMarioBros
* Strong Baseline: Getting averaging reward in 10 episodes over **3000** in SuperMarioBros
* Training Hint
  * The action should act ε-greedily
    * [x] Random action with probability ε
    * [ ] Also in testing
  * Linearly decline ε from 0.9 to some small value, say 0.05
    * [x] Decline per step
    * [x] Randomness is for exploration, agent is weak at start
  * Hyperparameters
    * [x] Rollout Storage Size: 10
    * [x] Perform Update Network Step: 10
    * [x] Process Number: 32
    * [x] Learning Rate: 7e-4

### 3.3. Training A2C on `gym-super-mario-bros`
1. Training A2C on all worlds: `$ python main.py --train_mario --world 0`
2. Training A2C on the world 1: `$ python main.py --train_mario --world 1`

### 3.4. Testing A2C on `gym-super-mario-bros`
1. Testing A2C trained on **all worlds** starting from **world 1 stage 1**:
    `$ python main.py --test_mario --do_render --world 0`
2. Testing A2C trained on the **world 1** starting from **world 1 stage 1**: 
    `$ python main.py --test_mario --do_render --world 1`
3. Testing A2C trained on **all worlds** for **all worlds**:
    `$ python test_mario.py --test_mario --do_render --world 0`
4. Testing A2C trained on the **world 1** for **all worlds**:
    `$ python test_mario.py --test_mario --do_render --world 1`

### 3.5. Rewards in 10 Episodes

### 3.6. Learning Curve
* Learning Curve of A2C
<img src="https://github.com/JasonYao81000/ADL2019/blob/master/hw3/results/reward_episode_a2c.png" width="100%">

###### tags: `NTU` `ADL` `2019`
