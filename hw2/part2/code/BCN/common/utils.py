import pickle
from collections import OrderedDict


def load_pkl(pkl_path):
    with open(pkl_path, mode='rb') as f:
        obj = pickle.load(f)

    return obj


class FixedOrderedDict(OrderedDict):
    """
    OrderedDict with fixed keys and decimal values.
    """
    def __init__(self, dictionary):
        self._dictionary = OrderedDict(dictionary)

    def __getitem__(self, key):
        return self._dictionary[key]

    def __setitem__(self, key, item):
        if key not in self._dictionary:
            raise KeyError(
                'FixedOrderedDict: The key \'{}\' is not defined.'.format(key))
        self._dictionary[key] = item

    def __str__(self):
        return ', '.join(['{}: {:8.5f}'.format(k, v)
                          for k, v in self._dictionary.items()])

    def get_dict(self):
        return self._dictionary
