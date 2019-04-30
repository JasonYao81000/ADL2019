import gym
from mario_env import create_mario_env
from a2c.vec_env.shmem_vec_env import ShmemVecEnv

def make_env(env_id, seed, rank):
    def _thunk():
        env = create_mario_env(env_id)
        env.seed(seed + rank)
        return env
    return _thunk

def make_vec_envs(env_name, seed, num_processes):
    envs = [
        make_env(env_name, seed, i)
        for i in range(num_processes)
    ]
    envs = ShmemVecEnv(envs, context='fork')
    return envs
