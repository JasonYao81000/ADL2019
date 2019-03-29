import torch
import torch.nn.functional as F
import math

class BiGruBatt4MaxFocalNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 similarity='CosineSimilarity'):
        super(BiGruBatt4MaxFocalNet, self).__init__()
        self.hidden_size = 128
        self.num_layers = 1
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        self.attn = torch.nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.v = torch.nn.Parameter(torch.rand(self.hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

        self.attention_rnn = torch.nn.GRU(self.hidden_size * 4, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.similarity = torch.nn.Linear(self.hidden_size * 2, 1, bias=False)
 
    def forward(self, context, context_lens, options, option_lens):
        batch_size = context.size(0)
        context_length = context.size(1)

        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        # context_outs: tensor of shape (B, Tc, H * 2)
        # context_h_n: tensor of shape (num_layers * 2, B, H)
        context_outs, context_h_n = self.rnn(context, h_0)

        # Sum bidirectional outputs.
        # context_outs: tensor of shape (B, Tc, H * 2) -> (B, Tc, H)
        context_outs = context_outs[:, :, :self.hidden_size] + context_outs[:, :, self.hidden_size:]

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_length = option.size(1)

            # Set initial hidden and cell states 
            h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # option_outs: tensor of shape (B, To, H * 2)
            # option_h_n: tensor of shape (num_layers * 2, B, H)
            option_outs, option_h_n = self.rnn(option, h_0)

            # Sum bidirectional outputs.
            # option_outs: tensor of shape (B, To, H * 2) -> (B, To, H)
            option_outs = option_outs[:, :, :self.hidden_size] + option_outs[:, :, self.hidden_size:]
            
            # repeat_context_outs: tensor of shape (B, Tc, H) -> (B, To, Tc, H)
            repeat_context_outs = torch.unsqueeze(context_outs, 1).repeat((1, option_length, 1, 1))
            # repeat_option_outs: tensor of shape (B, To, H) -> (B, To, Tc, H)
            repeat_option_outs = torch.unsqueeze(option_outs, 2).repeat((1, 1, context_length, 1))
            # attn_energies: tensor of shape (B, To, Tc, H * 2) -> (B, To, Tc, H)
            attn_energies = torch.tanh(self.attn(torch.cat((repeat_context_outs, repeat_option_outs), dim=-1)))
            # attn_energies: tensor of shape (B, To, Tc, H) -> (B, H, To * Tc)
            attn_energies = torch.reshape(attn_energies, (batch_size, option_length * context_length, self.hidden_size)).transpose(2, 1)
            # v: tensor of shape (B, 1, H)
            v = self.v.repeat(batch_size, 1).unsqueeze(1)
            # attn_energies: tensor of shape (B, 1, H) x (B, H, To x Tc) -> (B, 1, To x Tc)
            attn_energies = torch.bmm(v, attn_energies)
            # attn_energies: tensor of shape (B, 1, To x Tc) -> (B, To x Tc) -> (B, To, Tc)
            attn_energies = torch.reshape(attn_energies.squeeze(1), (batch_size, option_length, context_length))
            
            # attention_context: tensor of shape (B, Tc, To) x (B, To, H) -> (B, Tc, H)
            attention_context = torch.bmm(F.softmax(attn_energies, dim=1).transpose(1, 2), option_outs)
            # attention_option: tensor of shape (B, To, Tc) x (B, Tc, H) -> (B, To, H)
            attention_option = torch.bmm(F.softmax(attn_energies, dim=2), context_outs)

            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # attention_context_outs: tensor of shape (B, Tc, H * 2)
            # attention_context_h_n: tensor of shape (num_layers * 2, B, H)
            attention_context_outs, attention_context_h_n = self.attention_rnn(
                torch.cat((
                    context_outs, 
                    attention_context, 
                    torch.mul(context_outs, attention_context), 
                    context_outs - attention_context), dim=-1), h_1)

            # Sum bidirectional outputs.
            # attention_context_outs: tensor of shape (B, Tc, H * 2) -> (B, Tc, H)
            attention_context_outs = attention_context_outs[:, :, :self.hidden_size] + attention_context_outs[:, :, self.hidden_size:]

            # Max pooling over RNN outputs.
            # attention_context_outs_max: tensor of shape (B, H)
            attention_context_outs_max = attention_context_outs.max(dim=1)[0]

            # Set initial hidden and cell states 
            h_1 = torch.zeros(self.num_layers * 2, option.size(0), self.hidden_size).to(option.get_device())

            # Forward propagate RNN
            # attention_option_outs: tensor of shape (B, To, H * 2)
            # attention_option_h_n: tensor of shape (num_layers * 2, B, H)
            attention_option_outs, attention_option_h_n = self.attention_rnn(
                torch.cat((
                    option_outs, 
                    attention_option, 
                    torch.mul(option_outs, attention_option), 
                    option_outs - attention_option), dim=-1), h_1)
        
            # Sum bidirectional outputs.
            # attention_option_outs: tensor of shape (B, To, H * 2) -> (B, To, H)
            attention_option_outs = attention_option_outs[:, :, :self.hidden_size] + attention_option_outs[:, :, self.hidden_size:]

            # Max pooling over RNN outputs.
            # attention_option_outs_max: tensor of shape (batch, H)
            attention_option_outs_max = attention_option_outs.max(dim=1)[0]

            # Cosine similarity between context and each option.
            # logit = torch.nn.CosineSimilarity(dim=1)(attention_context_outs_max, attention_option_outs_max)
            logit = self.similarity(torch.cat((attention_context_outs_max, attention_option_outs_max), dim=-1))[:, 0]
            logits.append(logit)

        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
