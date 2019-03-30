import torch
torch.cuda.manual_seed_all(9487)
import torch.nn.functional as F

class BiGruLattMaxFocalNet(torch.nn.Module):
    """
    Args:
    """

    def __init__(self, dim_embeddings,
                 similarity='MLP'):
        super(BiGruLattMaxFocalNet, self).__init__()
        self.hidden_size = 128
        self.num_layers = 1
        self.rnn = torch.nn.GRU(dim_embeddings, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        self.attn = torch.nn.Linear(self.hidden_size * 4, 1)
        self.attention_rnn = torch.nn.GRU((self.hidden_size * 2) * 4, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        self.similarity = torch.nn.Linear((self.hidden_size * 2) * 2, 1, bias=False)
 
    def forward(self, context, context_lens, options, option_lens):
        batch_size = context.size(0)
        context_length = context.size(1)

        # Set initial hidden and cell states 
        h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())
        
        # Forward propagate RNN
        # context_outs: tensor of shape (B, Tc, H * 2)
        # context_h_n: tensor of shape (num_layers * 2, B, H)
        context_outs, context_h_n = self.rnn(context, h_0)

        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option_length = option.size(1)

            # Set initial hidden and cell states 
            h_0 = torch.zeros(self.num_layers * 2, context.size(0), self.hidden_size).to(context.get_device())

            # Forward propagate RNN
            # option_outs: tensor of shape (B, To, H * 2)
            # option_h_n: tensor of shape (num_layers * 2, B, H)
            option_outs, option_h_n = self.rnn(option, h_0)
            
            # repeat_context_outs: tensor of shape (B, Tc, H * 2) -> (B, To, Tc, H * 2)
            repeat_context_outs = torch.unsqueeze(context_outs, 1).expand(-1, option_length, -1, -1)
            # repeat_option_outs: tensor of shape (B, To, H * 2) -> (B, To, Tc, H * 2)
            repeat_option_outs = torch.unsqueeze(option_outs, 2).expand(-1, -1, context_length, -1)
            # attn_energies: tensor of shape (B, To, Tc, H * 4) -> (B, To, Tc)
            attn_energies = torch.tanh(self.attn(torch.cat((repeat_context_outs, repeat_option_outs), dim=-1))).squeeze(dim=-1)
            
            # attention_context: tensor of shape (B, Tc, To) x (B, To, H * 2) -> (B, Tc, H * 2)
            attention_context = torch.bmm(F.softmax(attn_energies, dim=1).transpose(1, 2), option_outs)
            # attention_option: tensor of shape (B, To, Tc) x (B, Tc, H * 2) -> (B, To, H * 2)
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

            # Max pooling over RNN outputs.
            # attention_context_outs_max: tensor of shape (B, H * 2)
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
            
            # Max pooling over RNN outputs.
            # attention_option_outs_max: tensor of shape (batch, H * 2)
            attention_option_outs_max = attention_option_outs.max(dim=1)[0]

            # Cosine similarity between context and each option.
            # logit = torch.nn.CosineSimilarity(dim=1)(attention_context_outs_max, attention_option_outs_max)
            logit = self.similarity(torch.cat((attention_context_outs_max, attention_option_outs_max), dim=-1))[:, 0]
            logits.append(logit)

        logits = F.softmax(torch.stack(logits, 1), dim=1)
        return logits
