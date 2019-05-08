def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--duel_dqn', action='store_true', help='whether DQN')
    parser.add_argument('--double_dqn', action='store_true', help='whether DQN')
    parser.add_argument('--ppo', action='store_true', help='whether PPO')
    return parser
