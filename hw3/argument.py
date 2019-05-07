def add_arguments(parser):
    '''
    Add your arguments here if needed. The TAs will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--double_dqn', action='store_true')
    parser.add_argument('--duel_dqn', action='store_true')
    parser.add_argument('--world', default=1, help='<world> is a number in {1, 2, 3, 4, 5, 6, 7, 8} indicating the world')
    parser.add_argument('--stage', default=1, help='<stage> is a number in {1, 2, 3, 4} indicating the stage within a world')
    parser.add_argument('--restore', default=None, help='the file name of the restoring model')
    return parser
