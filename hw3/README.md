# ADL HW3
Please don't revise test.py, environment.py,  atari_wrapper.py, mario_env.py, agent_dir/agent.py

## Installation
Type the following command to install OpenAI Gym Atari environment.

`$ pip3 install opencv-python gym gym[box2d] gym[atari]`

Please refer to [OpenAI's page](https://github.com/openai/gym) if you have any problem while installing.

2019/4/28 update:
If you encounter `AttributeError: module 'gym.envs.box2d' has no attribute 'LunarLander'`,
try to run `$ pip3 install gym[box2d]` again.

## How to run :
training policy gradient:
* `$ python3 main.py --train_pg`

testing policy gradient:
* `$ python3 test.py --test_pg`

training DQN:
* `$ python3 main.py --train_dqn`

testing DQN:
* `$ python3 test.py --test_dqn`

If you want to see your agent playing the game,
* `$ python3 test.py --test_[pg|dqn] --do_render`

## Bonus
Install SuperMarioBros: 

`$ pip3 install gym-super-mario-bros`

For more detail of this package, see:

https://github.com/Kautenja/gym-super-mario-bros

training SuperMarioBros:
* `$ python3 main.py --train_mario`

testing SuperMarioBros:
* `$ python3 test.py --test_mario`

## Code structure

```
.
├── agent_dir (all agents are placed here)
│   ├── agent.py (defined 4 required functions of the agent. DO NOT MODIFY IT)
│   ├── agent_dqn.py (DQN agent sample code)
│   ├── agent_pg.py (PG agent sample code)
│   └── agent_mario.py (Mario agent A2C sample code)
├── a2c (functions and classes used in A2C sample code)
│   ├── vec_env (code for vectorizing environment for A2C)
│   ├── actor_critic.py (define A2C model in pytorch)
│   ├── environment_a2c.py (process environment for A2C)
│   └── storage.py (define replay of A2C)
├── argument.py (you can add your arguments in here. we will use the default value when running test.py)
├── atari_wrapper.py (wrap the atari environment. DO NOT MODIFY IT)
├── environment.py (define the game environment in HW3, DO NOT MODIFY IT)
├── main.py (main function)
├── mario_env.py (define the mario environment. DO NOT MODIFY IT)
├── test.py (test script. we will use this script to test your agents. DO NOT MODIFY IT)

```
