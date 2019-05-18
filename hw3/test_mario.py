import argparse
import os
import numpy as np
from environment import Environment

seed = 9487

def parse():
    parser = argparse.ArgumentParser(description="Mario")
    parser.add_argument('--test_mario', action='store_true', help='whether test Mario')
    parser.add_argument('--video_dir', default=None, help='output video directory')
    parser.add_argument('--do_render', action='store_true', help='whether render environment')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def test(env_name, video_dir, total_episodes=10):
    from agent_dir.agent_mario import AgentMario
    print('Start playing %s ...' % (env_name))
    rewards = []
    args.video_dir = video_dir + '/' + env_name + '/'
    env = Environment(env_name, args, test=True)
    agent = AgentMario(env, args)
    env.seed(seed)
    for i in range(total_episodes):
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0
        
        #playing one game
        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            
        rewards.append(episode_reward)
    print('Env:', env_name, end=', ')
    print('Run %d episodes'%(total_episodes), end=', ')
    print('Mean:', np.mean(rewards))

def run(args):
    video_dir = args.video_dir
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)
    
    env_names = ["SuperMarioBros-%d-%d-v0" % (w + 1, s + 1) for w in range(8) for s in range(4)]
    rewards = []
    for env_name in env_names:
        test(env_name, video_dir)

if __name__ == '__main__':
    args = parse()
    run(args)
