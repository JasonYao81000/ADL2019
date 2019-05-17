def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--ppo', action='store_true')
    parser.add_argument('--double_dqn', action='store_true')
    parser.add_argument('--duel_dqn', action='store_true')
    parser.add_argument('--world', default=1, help='<world> is a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world, if `0` all world will be used in training')
    return parser
