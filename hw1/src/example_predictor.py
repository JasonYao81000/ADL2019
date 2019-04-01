import torch
torch.cuda.manual_seed_all(9487)
import logging
from base_predictor import BasePredictor
from modules import BiGruMaxFocalNet
from modules import BiGruBattMaxBCENet
from modules import BiGruBattMaxFocalNet
from modules import BiGruBNattMaxFocalNet
from modules import BiGruLattMaxFocalNet
from modules import BiGruLNattMaxFocalNet

from FocalLoss import FocalLoss

class ExamplePredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding,
                arch='BiGruMaxFocalNet', loss='FocalLoss',
                dropout_rate=0.2, margin=0, threshold=None,
                similarity='inner_product', **kwargs):
        super(ExamplePredictor, self).__init__(**kwargs)
        logging.info('building ' + arch + '...')
        if arch == 'BiGruMaxFocalNet': self.model = BiGruMaxFocalNet(embedding.size(1))
        if arch == 'BiGruBattMaxBCENet': self.model = BiGruBattMaxBCENet(embedding.size(1))
        if arch == 'BiGruBattMaxFocalNet': self.model = BiGruBattMaxFocalNet(embedding.size(1))
        if arch == 'BiGruBNattMaxFocalNet': self.model = BiGruBNattMaxFocalNet(embedding.size(1))
        if arch == 'BiGruLattMaxFocalNet': self.model = BiGruLattMaxFocalNet(embedding.size(1))
        if arch == 'BiGruLNattMaxFocalNet': self.model = BiGruLNattMaxFocalNet(embedding.size(1))
        
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
