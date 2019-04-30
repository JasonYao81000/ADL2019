"""

### NOTICE ###
You DO NOT need to upload this file

"""

import numpy as np
from collections import deque
import gym
from gym import spaces
from PIL import Image
import cv2

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


def _process_frame_mario(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    frame = np.expand_dims(frame, 0)
    return frame.astype(np.float32)

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        
        This object should only be converted to numpy array before being passed to the model.

        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(1, 84, 84), dtype=np.float32)
        self.status_order = {'small': 0, 'tall': 1, 'fireball': 2}
        self.prev_time = self.env.unwrapped._time
        self.prev_stat = self.status_order[self.env.unwrapped._player_status]
        self.prev_score = self.env.unwrapped._score
        self.prev_dist = self.env.unwrapped._x_position

    def step(self, action):
        ''' 
            Implementing custom rewards
                Time = -0.1
                Distance = +1 or 0 
                Player Status = +/- 5
                Score = 2.5 x [Increase in Score]
                Done = +50 [Game Completed] or -50 [Game Incomplete]
        '''
        obs, reward, done, info = self.env.step(action)


        reward = min(max((info['x_pos'] - self.prev_dist), 0), 2)
        self.prev_dist = info['x_pos']
        
        reward += (self.prev_time - info['time']) * -0.1
        self.prev_time = info['time']
        
        reward += (self.status_order[info['status']] - self.prev_stat) * 5
        self.prev_stat = self.status_order[info['status']]

        reward += (info['score'] - self.prev_score) * 0.025
        self.prev_score = info['score']

        if done:
            if info['life'] != 255:
                reward += 50
            else:
                reward -= 50

        return _process_frame_mario(obs), reward, done, info

    def reset(self):
        obs = _process_frame_mario(self.env.reset())
        self.prev_time = self.env.unwrapped._time
        self.prev_stat = self.status_order[self.env.unwrapped._player_status]
        self.prev_score = self.env.unwrapped._score
        self.prev_dist = self.env.unwrapped._x_position
        return obs

    def change_level(self, level):
        self.env.change_level(level)


class BufferSkipFrames(gym.Wrapper):
    def __init__(self, env=None, skip=4, shape=(84, 84)):
        super(BufferSkipFrames, self).__init__(env)
        self.counter = 0
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(4, 84, 84), dtype=np.float32)
        self.skip = skip
        self.buffer = deque(maxlen=self.skip)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        counter = 1
        total_reward = reward
        self.buffer.append(obs)

        for i in range(self.skip - 1):            
            if not done:
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                counter +=1
                self.buffer.append(obs)
            else:
                self.buffer.append(obs)
        frame = LazyFrames(list(self.buffer))
        #frame = np.stack(self.buffer, axis=0)
        #frame = np.reshape(frame, (4, 84, 84))
        return frame, total_reward, done, info

    def reset(self):
        self.buffer.clear()
        obs = self.env.reset()
        for i in range(self.skip):
            self.buffer.append(obs)

        frame = np.stack(self.buffer, axis=0)
        frame = np.reshape(frame, (4, 84, 84))
        return frame
    
    def change_level(self, level):
        self.env.change_level(level)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        if observation is not None:    # for future meta implementation
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

            unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
            unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

            return (observation - unbiased_mean) / (unbiased_std + 1e-8)
        
        else:
            return observation

    def change_level(self, level):
        self.env.change_level(level)

def wrap_mario(env):
    env = ProcessFrameMario(env)
    env = NormalizedEnv(env)
    env = BufferSkipFrames(env)
    return env

def create_mario_env(env_id):
    env = gym_super_mario_bros.make(env_id)
    env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    return env
