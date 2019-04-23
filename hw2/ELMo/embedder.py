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
        if self.n_ctx_embs == 1 and self.ctx_emb_dim == 4096:
            self.stacked_embeddings = StackedEmbeddings([
                FlairEmbeddings('news-forward'),        # 2048
                FlairEmbeddings('news-backward'),       # 2048
            ])
        elif self.n_ctx_embs == 3 and self.ctx_emb_dim == 1024:
            self.stacked_embeddings = StackedEmbeddings([
                ELMoEmbeddings('original'),             # 3072
            ])
        elif self.n_ctx_embs == 4 and self.ctx_emb_dim == 1024:
            self.stacked_embeddings = StackedEmbeddings([
                # BertEmbeddings('bert-large-cased'),     # 4096
                BertEmbeddings('bert-large-uncased'),     # 4096
            ])
        elif self.n_ctx_embs == 7 and self.ctx_emb_dim == 2048:
            self.stacked_embeddings = StackedEmbeddings([
                # BertEmbeddings('bert-large-cased'),     # 4096
                BertEmbeddings('bert-large-uncased'),     # 4096
                ELMoEmbeddings('original'),             # 3072
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
        if self.n_ctx_embs == 1 and self.ctx_emb_dim == 4096:
            for i, sentence in enumerate(Sentences):
                for j, token in enumerate(sentence):
                    results[i, j, 0, :] = token.embedding
        elif self.n_ctx_embs == 3 and self.ctx_emb_dim == 1024:
            for i, sentence in enumerate(Sentences):
                for j, token in enumerate(sentence):
                    results[i, j, 0, :] = token.embedding[0:1024]
                    results[i, j, 1, :] = token.embedding[1024:2048]
                    results[i, j, 2, :] = token.embedding[2048:3072]
        elif self.n_ctx_embs == 4 and self.ctx_emb_dim == 1024:
            for i, sentence in enumerate(Sentences):
                for j, token in enumerate(sentence):
                    results[i, j, 0, :] = token.embedding[0:1024]
                    results[i, j, 1, :] = token.embedding[1024:2048]
                    results[i, j, 2, :] = token.embedding[2048:3072]
                    results[i, j, 3, :] = token.embedding[3072:4096]
        elif self.n_ctx_embs == 7 and self.ctx_emb_dim == 2048:
            for i, sentence in enumerate(Sentences):
                for j, token in enumerate(sentence):
                    results[i, j, 0, :1024] = token.embedding[0:1024]
                    results[i, j, 1, :1024] = token.embedding[1024:2048]
                    results[i, j, 2, :1024] = token.embedding[2048:3072]
                    results[i, j, 3, :1024] = token.embedding[3072:4096]
                    results[i, j, 4, 1024:2048] = token.embedding[4096:5120]
                    results[i, j, 5, 1024:2048] = token.embedding[5120:6144]
                    results[i, j, 6, 1024:2048] = token.embedding[6144:7168]
            
        return results
