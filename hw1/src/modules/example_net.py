import torch


class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        self.hidden_size = 256
        self.num_layers = 1

        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        # self.similarity = torch.nn.Sequential(
        #     torch.nn.Linear((self.num_layers * 2 * self.hidden_size) * 2, self.hidden_size),     # 2 for context and option
        #     # torch.nn.BatchNorm1d(self.hidden_size),
        #     torch.nn.ReLU(),
        #     # torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
        #     # torch.nn.BatchNorm1d(self.hidden_size // 2),
        #     # torch.nn.ReLU(),
        #     torch.nn.Linear(self.hidden_size, 1),
        #     torch.nn.Sigmoid()
        # )
        # # use the modules apply function to recursively apply the initialization
        # self.similarity.apply(self.init_normal)
        
        self.probability = torch.nn.Softmax(dim=1)

        # self.linear = torch.nn.Sequential(
        #     torch.nn.Linear(self.hidden_size * 2, self.hidden_size),           # 2 for bidirectional
        #     torch.nn.BatchNorm1d(self.hidden_size),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.hidden_size, self.hidden_size // 2),
        #     torch.nn.BatchNorm1d(self.hidden_size // 2),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(self.hidden_size // 2, 1),
        #     torch.nn.Sigmoid()
        # )
        # # use the modules apply function to recursively apply the initialization
        # self.linear.apply(self.init_normal)

        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(dim_embeddings, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256)
        # )

    # # initialization function, first checks the module type,
    # # then applies the desired changes to the weights
    # def init_normal(self, m):
    #     if type(m) == torch.nn.Linear:
    #         torch.nn.init.uniform_(m.weight)

    def forward(self, context, context_lens, options, option_lens):
        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        out0, h_n_0 = self.rnn(context, h_0)     # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
        
        # # Decode the hidden state of the last time step
        # context = self.linear(out0[:, -1, :])
        # context = self.mlp(context).max(1)[0]
        # context_max = out0.max(0)[0]
        context_h = h_n_0.transpose(1, 0)
        context_h = context_h.contiguous().view(context.size(0), -1)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, option.size(0), self.hidden_size).to(option.get_device())

            # Forward propagate RNN
            out1, h_n_1 = self.rnn(option, h_1)  # out: tensor of shape (batch_size, seq_length, hidden_size * 2)
            # option_max = out1.max(1)[0]
            option_h = h_n_1.transpose(1, 0)
            option_h = option_h.contiguous().view(option.size(0), -1)

            # # Decode the hidden state of the last time step
            # option = self.linear(out1[:, -1, :])
            # option = self.mlp(option).max(1)[0]

            # logit = ((context_h - option_h) ** 2).sum(-1)
            logit = torch.nn.CosineSimilarity(dim=1)(context_h, option_h)
            # logit = torch.squeeze(self.similarity(torch.cat((context, option), dim=1)))
            # logit = self.similarity(torch.cat((context_h, option_h), dim=1))[:, 0]
            # logit = torch.squeeze(self.linear(out1[:, -1, :]))
            logits.append(logit)
        
        # logits = torch.stack(logits, 1)
        logits = self.probability(torch.stack(logits, 1))
        return logits
