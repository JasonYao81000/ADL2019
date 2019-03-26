import torch
import torch.nn.functional as F

class GruCosMeanNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 similarity='CosineSimilarity'):
        super(GruCosMeanNet, self).__init__()
        self.hidden_size = 256
        self.num_layers = 1
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

    def forward(self, context, context_lens, options, option_lens):
        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        # context_outs: tensor of shape (batch, context_length, hidden_size * 2)
        # context_h_n: tensor of shape (num_layers * 2, batch, hidden_size)
        context_outs, context_h_n = self.rnn(context, h_0)

        # Mean pooling over RNN outputs.
        # context_outs_mean: tensor of shape (batch, hidden_size * 2)
        context_outs_mean = context_outs.mean(dim=1)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # Set initial hidden and cell states 
            h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
            
            # Forward propagate RNN
            # option_outs: tensor of shape (batch, option_length, hidden_size * 2)
            # option_h_n: tensor of shape (num_layers * 2, batch, hidden_size)
            option_outs, option_h_n = self.rnn(option, h_0)

            # Mean pooling over RNN outputs.
            # option_outs_mean: tensor of shape (batch, hidden_size * 2)
            option_outs_mean = option_outs.mean(dim=1)

            # Cosine similarity between context and each option.
            logit = torch.nn.CosineSimilarity(dim=1)(context_outs_mean, option_outs_mean)
            logits.append(logit)
        
        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
