import torch
import logging
from base_predictor import BasePredictor
# from modules import ExampleNet
# from modules import GruCosNet
# from modules import BahdanauAttentionsMaxNet
# from modules import BahdanauAttentionsMaxFocalNet
from modules import BahdanauNormAttentionsMaxFocalNet
# from modules import BiGruBattMaxFocalNet
# from modules import BiGruBatt4MaxNet
# from modules import BiGruBatt4MaxFocalNet

from FocalLoss import FocalLoss

class ExamplePredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding,
                 dropout_rate=0.2, loss='FocalLoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(ExamplePredictor, self).__init__(**kwargs)
        logging.info('building BahdanauNormAttentionsMaxFocalNet...')
        self.model = BahdanauNormAttentionsMaxFocalNet(embedding.size(1),
                                similarity=similarity)
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)

        # use cuda
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'FocalLoss': FocalLoss(),
            'BCELoss': torch.nn.BCEWithLogitsLoss()
        }[loss]
        logging.info('using ' + loss + '...')

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        return logits
