import numpy as np
# from .elmo import Embedder as ELMoEmbedder
from flair.data import Sentence
# from flair.embeddings import WordEmbeddings
from flair.embeddings import FlairEmbeddings
from flair.embeddings import ELMoEmbeddings
from flair.embeddings import BertEmbeddings
from flair.embeddings import StackedEmbeddings

class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        # TODO
        # self.e = ELMoEmbedder('./ELMo/output')
        # create a StackedEmbedding object that combines glove and forward/backward flair embeddings
        self.stacked_embeddings = StackedEmbeddings([
            # WordEmbeddings('glove'),              # 100
            FlairEmbeddings('news-forward'),        # 2048
            FlairEmbeddings('news-backward'),       # 2048
            ELMoEmbeddings('original'),             # 3072
            BertEmbeddings('bert-large-uncased'),   # 4096
        ])

    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        # TODO
        results = np.zeros((len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim), dtype=np.float32)
        # embeddings = self.e.sents2elmo(sentences, output_layer=-2)
        # for i, embedding in enumerate(embeddings):
        #     embedding = np.transpose(embedding, (1, 0, 2))
        #     results[i, :embedding.shape[0], :, :] = embedding

        # Create sentences
        Sentences = [Sentence(' '.join(x)) for x in sentences]
        # embed words in sentence
        self.stacked_embeddings.embed(Sentences)
        for i, sentence in enumerate(Sentences):
            for j, token in enumerate(sentence):
                results[i, j, 0, :2048] = token.embedding[0:2048]
                results[i, j, 1, :2048] = token.embedding[2048:4096]
                results[i, j, 2, 2048:3072] = token.embedding[4096:5120]
                results[i, j, 3, 2048:3072] = token.embedding[5120:6144]
                results[i, j, 4, 2048:3072] = token.embedding[6144:7168]
                results[i, j, 5, 3072:4096] = token.embedding[7168:8192]
                results[i, j, 6, 3072:4096] = token.embedding[8192:9216]
                results[i, j, 7, 3072:4096] = token.embedding[9216:10240]
                results[i, j, 8, 3072:4096] = token.embedding[10240:11264]
            
        return results
