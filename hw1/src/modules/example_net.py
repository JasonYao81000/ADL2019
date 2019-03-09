import torch


class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product'):
        super(ExampleNet, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim_embeddings, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256)
        )

    def forward(self, context, context_lens, options, option_lens):
        context = self.mlp(context).max(1)[0]
        logits = []
        for i, option in enumerate(options.transpose(1, 0)):
            option = self.mlp(option).max(1)[0]
            logit = ((context - option) ** 2).sum(-1)
            logits.append(logit)
        logits = torch.stack(logits, 1)
        return logits
