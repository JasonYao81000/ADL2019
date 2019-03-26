import torch
import numpy as np

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
        
        self.rnn1 = torch.nn.GRU(dim_embeddings, 128, 1, bidirectional=True, batch_first=True)
        self.rnn2 = torch.nn.GRU(256, 256, 1, bidirectional=True, batch_first=True)

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
#        self.projection = torch.nn.Linear(43200,100)

    def forward(self, context, context_lens, options, option_lens):
#        print(options)
        h_0 = torch.zeros(2, context.size(0), 128).to(context.get_device())
        context_out, context_hidden = self.rnn1(context, h_0)
        #context_out = self.mlp(context_out).max(1)[0]
#        print(context_out.size())
        context_h = context_hidden.transpose(1,0)
        context_h = context_h.contiguous().view(context.size(0),-1)
        #print(context_out.shape)
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            attenArr = []
            h_1 = torch.zeros(2, context.size(0), 128).to(context.get_device())
            option_out, option_hidden = self.rnn1(option, h_1)
#            print(option_out.size())
            option_h = option_hidden.transpose(1,0)
            option_h = option_h.contiguous().view(option.size(0),-1)
            #option_out = self.mlp(option_out).max(1)[0]
            #logit = ((context_out - option_out) ** 2).sum(-1)   
            for j in range(len(context_out[0,:,0])):
                atten = []
                for n in range(len(option_out[0,:,0])):
                    logi = torch.nn.CosineSimilarity(dim=1)(context_out[:,j,:], option_out[:,n,:])
                    atten.append(logi.tolist())
                attenArr.append(atten)

            attenArr = torch.tensor(attenArr).cuda()
            attenCon = attenArr.transpose(2,0)@context_out
            attenOp = attenArr.transpose(2,1).transpose(1,0)@option_out
            attenCon_out, attenCon_hidden = self.rnn2(attenCon)
            attenOp_out, attenOp_hidden = self.rnn2(attenOp)          
            attenCon_h = attenCon_hidden.transpose(1,0)
            attenCon_h = attenCon_h.contiguous().view(attenCon.size(0),-1)         
            attenOp_h = attenOp_hidden.transpose(1,0)
            attenOp_h = attenOp_h.contiguous().view(attenOp.size(0),-1)
#            for j in range(len(attenArr[0][:])):
#                attenval = []
#                for n in range(len(attenArr[:][0])):
#                    print(attenArr[n][j])
#                    print(context_out[:,n,:].size())
#                    v1 = attenArr[n][j]
#                    v2 = context_out[:,n,:]
#                    attenval = attenval + torch.mm(v2,v1)
#                atten_con.append(attenval)
#            for j in range(len(attenArr[:,0])):
#                atten_op.append(sum(attenArr[j,:]*option_out.detach().cpu().numpy()))
#            print(np.array(atten_con).shape)
#            print(np.array(atten_op).shape)
            logit = torch.nn.CosineSimilarity(dim=1)(attenCon_h, attenOp_h)
#            logit = self.s(logit + option_h).contiguous().view(43200,-1)
#            print(logit.size())
#            logits.append(self.projection(logit))
            logits.append(logit)
        logits = self.s(torch.stack(logits, 1))
#        print(logits.size())
        
#        print(logits.size())
        #return logits
        return logits
