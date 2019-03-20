import torch

class GruCosNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 similarity='CosineSimilarity'):
        super(GruCosNet, self).__init__()
        self.hidden_size = 256
        self.num_layers = 1
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)        
        self.probability = torch.nn.Softmax(dim=1)

    def forward(self, context, context_lens, options, option_lens):
        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        out0, h_n_0 = self.rnn(context, h_0)
        
        # Flatten the hidden states of the all time steps
        context_h = h_n_0.transpose(1, 0)
        context_h = context_h.contiguous().view(context.size(0), -1)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, option.size(0), self.hidden_size).to(option.get_device())

            # Forward propagate RNN
            # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
            out1, h_n_1 = self.rnn(option, h_1)

            # Flatten the hidden states of the all time steps
            option_h = h_n_1.transpose(1, 0)
            option_h = option_h.contiguous().view(option.size(0), -1)
            
            # Cosine similarity between context and each option.
            logit = torch.nn.CosineSimilarity(dim=1)(context_h, option_h)
            logits.append(logit)
        
        logits = self.probability(torch.stack(logits, 1))
        return logits
