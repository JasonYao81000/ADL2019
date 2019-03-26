import torch
import torch.nn.functional as F

class GruAttCosMeanNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 similarity='CosineSimilarity'):
        super(GruAttCosMeanNet, self).__init__()
        self.hidden_size = 256
        self.num_layers = 1
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        self.key_layer = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.query_layer = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        self.energy_layer = torch.nn.Linear(self.hidden_size, 1, bias=False)
        self.attention_rnn = torch.nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
 
    def forward(self, context, context_lens, options, option_lens):
        context_length = context.size(1)

        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        # context_outs: tensor of shape (batch, context_length, hidden_size * 2)
        # context_h_n: tensor of shape (num_layers * 2, batch, hidden_size)
        context_outs, context_h_n = self.rnn(context, h_0)

        # context_key: tensor of shape (batch, context_length, hidden_size)
        context_key = self.key_layer(context_outs)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_length = option.size(1)

            # Set initial hidden and cell states 
            h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # option_outs: tensor of shape (batch, option_length, hidden_size * 2)
            # option_h_n: tensor of shape (num_layers * 2, batch, hidden_size)
            option_outs, option_h_n = self.rnn(option, h_0)

            # option_query: tensor of shape (batch, option_length, hidden_size)
            option_query = self.query_layer(option_outs)
            
            # repeat_context_key: tensor of shape (batch, context_length, hidden_size) -> (batch, option_length, context_length, hidden_size)
            repeat_context_key = torch.unsqueeze(context_key, 1).repeat((1, option_length, 1, 1))
            # repeat_option_query: tensor of shape (batch, option_length, hidden_size) -> (batch, option_length, context_length, hidden_size)
            repeat_option_query = torch.unsqueeze(option_query, 2).repeat((1, 1, context_length, 1))
            # attentions: tensor of shape (batch, option_length, context_length, hidden_size) -> (batch, option_length, context_length)
            attentions = self.energy_layer(torch.tanh(repeat_option_query + repeat_context_key)).squeeze(dim=-1)
            
            # attention_context: tensor of shape (batch, context_length, option_length) x (batch, option_length, hidden_size) -> (batch, context_length, hidden_size)
            attention_context = torch.bmm(F.softmax(attentions, dim=1).transpose(1, 2), option_query)
            # attention_option: tensor of shape (batch, option_length, context_length) x (batch, context_length, hidden_size) -> (batch, option_length, hidden_size)
            attention_option = torch.bmm(F.softmax(attentions, dim=2), context_key)

            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # attention_context_outs: tensor of shape (batch, context_length, hidden_size * 2)
            # attention_context_h_n: tensor of shape (num_layers * 2, batch, hidden_size)
            attention_context_outs, attention_context_h_n = self.attention_rnn(attention_context, h_1)

            # Mean pooling over RNN outputs.
            # attention_context_outs_mean: tensor of shape (batch, hidden_size * 2)
            attention_context_outs_mean = attention_context_outs.mean(dim=1)

            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, option.size(0), self.hidden_size).to(option.get_device())

            # Forward propagate RNN
            # attention_option_outs: tensor of shape (batch, option_length, hidden_size * 2)
            # attention_option_h_n: tensor of shape (num_layers * 2, batch, hidden_size)
            attention_option_outs, attention_option_h_n = self.attention_rnn(attention_option, h_1)

            # Mean pooling over RNN outputs.
            # attention_option_outs_mean: tensor of shape (batch, hidden_size * 2)
            attention_option_outs_mean = attention_option_outs.mean(dim=1)

            # Cosine similarity between context and each option.
            logit = torch.nn.CosineSimilarity(dim=1)(attention_context_outs_mean, attention_option_outs_mean)
            logits.append(logit)

        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
