import torch
torch.cuda.manual_seed_all(9487)
import logging
from base_predictor import BasePredictor
from modules import BiGruMaxFocalNet
from modules import BiGruBattMaxBCENet
from modules import BiGruBattMaxFocalNet
from modules import BiGruBattMeanFocalNet
from modules import BiGruBatt5MaxFocalNet
from modules import BiGruBattDropMaxFocalNet
from modules import BiGruBNattMaxFocalNet
from modules import BiGruLattMaxFocalNet
from modules import BiGruLNattMaxFocalNet
from modules import BiLstmBattMaxFocalNet
from modules import DeepBiGruBattMaxFocalNet
from modules import FatBiGruBattMaxFocalNet

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
        self.arch = arch
        logging.info('building ' + self.arch + '...')
        if self.arch == 'BiGruMaxFocalNet': self.model = BiGruMaxFocalNet(embedding.size(1))
        if self.arch == 'BiGruBattMaxBCENet': self.model = BiGruBattMaxBCENet(embedding.size(1))
        if self.arch == 'BiGruBattMaxFocalNet': self.model = BiGruBattMaxFocalNet(embedding.size(1))
        if self.arch == 'BiGruBattMeanFocalNet': self.model = BiGruBattMeanFocalNet(embedding.size(1))
        if self.arch == 'BiGruBatt5MaxFocalNet': self.model = BiGruBatt5MaxFocalNet(embedding.size(1))
        if self.arch == 'BiGruBNattMaxFocalNet': self.model = BiGruBNattMaxFocalNet(embedding.size(1))
        if self.arch == 'BiGruLattMaxFocalNet': self.model = BiGruLattMaxFocalNet(embedding.size(1))
        if self.arch == 'BiGruLNattMaxFocalNet': self.model = BiGruLNattMaxFocalNet(embedding.size(1))
        if self.arch == 'BiGruBattDropMaxFocalNet': self.model = BiGruBattDropMaxFocalNet(embedding.size(1), dropout_rate=dropout_rate)
        if self.arch == 'BiLstmBattMaxFocalNet': self.model = BiLstmBattMaxFocalNet(embedding.size(1))
        if self.arch == 'DeepBiGruBattMaxFocalNet': self.model = DeepBiGruBattMaxFocalNet(embedding.size(1), dropout_rate=dropout_rate)
        if self.arch == 'FatBiGruBattMaxFocalNet': self.model = FatBiGruBattMaxFocalNet(embedding.size(1))

        if self.arch[:3] != 'Emb':
            self.embedding = torch.nn.Embedding(embedding.size(0),
                                                embedding.size(1))
            self.embedding.weight = torch.nn.Parameter(embedding)
            self.embedding = self.embedding.to(self.device)

        # use cuda
        self.model = self.model.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        self.loss = {
            'FocalLoss': FocalLoss(),
            'BCELoss': torch.nn.BCEWithLogitsLoss()
        }[loss]
        logging.info('using ' + loss + '...')

    def _run_iter(self, batch, training):
        if self.arch[:3] == 'Emb':
            logits = self.model.forward(
                batch['context'].to(self.device),
                batch['context_lens'],
                batch['options'].to(self.device),
                batch['option_lens'])
        else:
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
        if self.arch[:3] == 'Emb':
            logits = self.model.forward(
                batch['context'].to(self.device),
                batch['context_lens'],
                batch['options'].to(self.device),
                batch['option_lens'])
        else:
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
            logits = self.model.forward(
                context.to(self.device),
                batch['context_lens'],
                options.to(self.device),
                batch['option_lens'])
        return logits
