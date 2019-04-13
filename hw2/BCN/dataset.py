import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Part1Dataset(Dataset):
    def __init__(self, data, word_vocab, char_vocab):
        self._data = [{
            'Id': d['Id'],
            'text_orig': d['text'],
            'text_word': [word_vocab.vtoi(w.lower()) for w in d['text']],
            'text_char': [[char_vocab.vtoi(c) for c in w] for w in d['text']],
            'label': int(d['label']) - 1
        } for d in tqdm(data, desc='[*] Indexizing', dynamic_ncols=True)]

        token_cnt = oov_cnt = 0
        unk_idx = word_vocab.sp.unk.idx
        for d in tqdm(self._data, desc='[*] Calculating OOV rate', dynamic_ncols=True):
            for t in d['text_word']:
                token_cnt += 1
                if t == unk_idx:
                    oov_cnt += 1
        print('[#] Word OOV rate: {}/{} ='.format(oov_cnt, token_cnt),
              '{:.3f}%'.format(oov_cnt/token_cnt * 100))

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)


def create_collate_fn(word_vocab, char_vocab, max_sent_len, max_word_len):
    word_pad_idx = word_vocab.sp.pad.idx
    char_pad_idx = char_vocab.sp.pad.idx

    # This recursive version can account of arbitrary depth. However, the required stack
    # allocation may harm performance.
    # def pad(batch, max_len, padding):
    #     l, p = max_len[0], padding[0]
    #     for i, b in enumerate(batch):
    #         batch[i] = b[:l]
    #         batch[i] += [[p] for _ in range(l - len(b))]
    #         if len(max_len) > 1:
    #             batch[i] = pad(batch[i], max_len[1:], padding[1:])
    #
    #     return batch

    def pad(batch, max_len, padding, depth=1):
        for i, b in enumerate(batch):
            if depth == 1:
                batch[i] = b[:max_len]
                batch[i] += [padding for _ in range(max_len - len(b))]
            elif depth == 2:
                for j, bb in enumerate(b):
                    batch[i][j] = bb[:max_len]
                    batch[i][j] += [padding] * (max_len - len(bb))

        return batch

    def collate_fn(batch):
        Id = [b['Id'] for b in batch]
        text_orig = [b['text_orig'] for b in batch]
        text_word = [b['text_word'] for b in batch]
        text_char = [b['text_char'] for b in batch]
        label = [b['label'] for b in batch]

        max_len = min(max(map(len, text_word)), max_sent_len)
        text_word = pad(text_word, max_len, word_pad_idx)
        text_char = pad(text_char, max_len, [char_pad_idx])
        max_len = min(np.max([[len(w) for w in s] for s in text_char]), max_word_len)
        text_char = pad(text_char, max_len, char_pad_idx, depth=2)

        text_word = torch.tensor(text_word)
        text_char = torch.tensor(text_char)
        text_pad_mask = text_word != word_pad_idx
        label = torch.tensor(label)

        return {
            'Id': Id,
            'text_orig': text_orig,
            'text_word': text_word,
            'text_char': text_char,
            'text_pad_mask': text_pad_mask,
            'label': label
        }

    return collate_fn


def create_data_loader(dataset, word_vocab, char_vocab, max_sent_len, max_word_len,
                       batch_size, n_workers, shuffle=True):
    collate_fn = create_collate_fn(word_vocab, char_vocab, max_sent_len, max_word_len)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers,
        collate_fn=collate_fn)

    return data_loader
