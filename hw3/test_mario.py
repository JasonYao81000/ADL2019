import argparse
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

def run(args):
    from agent_dir.agent_mario import AgentMario
    env_names = ["SuperMarioBros-%d-%d-v0" % (w + 1, s + 1) for w in range(8) for s in range(4)]
    rewards = []
    for env_name in env_names:
        print('Start playing %s ...' % (env_name))
        env = Environment(env_name, args, test=True)
        agent = AgentMario(env, args)
        env.seed(seed)
        state = env.reset()
        agent.init_game_setting()
        done = False
        episode_reward = 0.0

        while(not done):
            action = agent.make_action(state, test=True)
            state, reward, done, info = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        print('Reward:', episode_reward)

    print('Run %d envs' % (len(env_names)))
    print('Mean:', np.mean(rewards))

if __name__ == '__main__':
    args = parse()
    run(args)
