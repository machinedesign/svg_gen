import torch.nn as nn
import torch

class Model(nn.Module):
    def __init__(self, input_size=32, input_repr_size=200, vocab_size=10, emb_size=128, hidden_size=128, num_layers=1, nb_features=1, use_cuda=False):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.X = None
        self.input_repr = nn.Sequential(
            nn.Conv2d(1, 64, 9),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 9),
            nn.ReLU(True),
            nn.MaxPool2d(20, 20),
        )
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.out_token  = nn.Linear(hidden_size + input_repr_size, vocab_size)
        self.out_value = nn.Linear(hidden_size + input_repr_size, nb_features)
    
    def given(self, X):
        self.X = X

    def forward(self, inp):
        x = self.emb(inp)
        o, _ = self.lstm(x)
        o = o.contiguous()
        X = self.X
        
        X = X.view(X.size(0), 1, 64, 64)
        X = self.input_repr(X)
        X = X.view(X.size(0), -1)

        X = X.view(X.size(0), 1, X.size(1))
        X = X.repeat(1, o.size(1), 1)
        o = torch.cat((o, X), 2)
        o = o.view(o.size(0) * o.size(1), o.size(2))
        o = self.out_token(o)
        return o

    def next_token(self, inp, state):
        if self.use_cuda:
            inp = inp.cuda()
        x = self.emb(inp)
        _, state = self.lstm(x, state)
        h, c = state
        h = h.view(h.size(0) * h.size(1), h.size(2))
        X = self.X
        
        X = X.view(X.size(0), 1, 64, 64)
        X = self.input_repr(X)
        X = X.view(X.size(0), -1)


        X = X.repeat(h.size(0), 1)
        h = torch.cat((h, X), 1)
        o = self.out_token(h)
        return o, state
