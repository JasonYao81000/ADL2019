# ADL HW3

## Installation :
```
$ sudo pip install opencv-python

$ sudo pip install gym[box2d]

$ sudo pip install gym[atari]

$ sudo pip install gym-super-mario-bros
```


## How to run :

testing policy gradient:
```
$ python3.7 test.py --test_pg
```

testing policy gradient (Proximal Policy Optimization):
```
$ python3.7 test.py --test_pg --ppo
```

testing DQN:
```
$ python3.7 test.py --test_dqn
```

testing dueling DQN:
```
$ python3.7 test.py --test_dqn --duel_dqn
```

testing double DQN:
```
$ python3.7 test.py --test_dqn --double_dqn
```

testing dueling + double DQN:
```
$ python3.7 test.py --test_dqn --duel_dqn --double_dqn
```

testing mario:
```
$ python3.7 test.py --test_mario
```
