import torch


class ExampleNet(torch.nn.Module):
    """

    Args:

    """
#    def __init__(self, hidden_size, out_size, n_layers=1, batch_size=1):
#        super(EncoderRNNWithVector, self).__init__()
#
#        self.batch_size = bactch_size
#        self.hidden_size = hidden_size
#        self.n_layers = n_layers
#        self.out_size = out_size
#
#        # 这里指定了 BATCH FIRST
#        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, batch_first=True)
#
#       # 加了一个线性层，全连接
#        self.out = torch.nn.Linear(hidden_size, out_size)
#        
#    def forward(self, word_inputs, hidden):
#
#        # -1 是在其他确定的情况下，PyTorch 能够自动推断出来，view 函数就是在数据不变的情况下重新整理数据维度
#        # batch, time_seq, input
#        inputs = word_inputs.view(self.batch_size, -1, self.hidden_size)
#
#       # hidden 就是上下文输出，output 就是 RNN 输出
#        output, hidden = self.gru(inputs, hidden)
#
#        output = self.out(output)
#
#        # 仅仅获取 time seq 维度中的最后一个向量
#        # the last of time_seq
#        output = output[:,-1,:]
#
#        return output, hidden
#
#    def init_hidden(self):
#
#        # 这个函数写在这里，有一定迷惑性，这个不是模型的一部分，是每次第一个向量没有上下文，在这里捞一个上下文，仅此而已。
#        hidden = torch.autograd.Variable(torch.zeros(2, 300, 216))
#        return hidden


    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        
        self.gru = torch.nn.GRU(dim_embeddings, 108, bidirectional=True, batch_first=True)
        
#        self.mlp = torch.nn.Sequential(
##            torch.nn.GRU(input_size=dim_embeddings, hidden_size=216, num_layers=1, bidirectional=True),
#            torch.nn.Linear(216, 350),
#            torch.nn.ReLU(),
#            torch.nn.Dropout(0.4),            
#            torch.nn.Linear(350, 450),
#            torch.nn.ReLU(),
#            torch.nn.Dropout(0.4),
#            torch.nn.Linear(450, 612),
#            torch.nn.ReLU(),
#            torch.nn.Dropout(0.2),
#            torch.nn.Linear(612, 100),
#            torch.nn.Sigmoid()
#        )
        self.s = torch.nn.Softmax(dim=1)

    def forward(self, context, context_lens, options, option_lens):
#        print(context)
#        print(options)
        context_out, context_hidden = self.gru(context)
        #context_out = self.mlp(context_out).max(1)[0]
        context_h = context_hidden.transpose(1,0)
        context_h = context_h.contiguous().view(context.size(0),-1)
        #print(context_out.shape)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_out, option_hidden = self.gru(option)
            option_h = option_hidden.transpose(1,0)
            option_h = option_h.contiguous().view(context.size(0),-1)
            #option_out = self.mlp(option_out).max(1)[0]
            #logit = ((context_out - option_out) ** 2).sum(-1)    
            logit = torch.nn.CosineSimilarity(dim=1)(context_h, option_h)       
            logits.append(logit)
        logits = torch.stack(logits, 1)
        #return logits
        return self.s(logits)
