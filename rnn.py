import torch.nn as nn

class Sim(nn.Module):

    def __init__(self, img_size=64, vocab_size=10, emb_size=128, hidden_size=128, num_layers=1):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.img_size = img_size
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.enc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out_img = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, img_size * img_size)
        )
    
    def forward(self, inp):
        x = self.emb(inp)
        o, _ = self.rnn(x)
        o = o[:, -1, :]
        o = o.contiguous()
        h = self.enc(o)
        o = self.out_img(h)
        return o, h


class Model(nn.Module):

    def __init__(self, img_size=64, input_size=32, input_repr_size=200, vocab_size=10, emb_size=128, 
                 hidden_size=128, num_layers=1, nb_features=1, use_cuda=False):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.input_size = input_size
        self.num_layers = num_layers
        self.img_size = img_size
        self.X = None
        self.input_repr = nn.Sequential(
            nn.Conv2d(1, 128, 9),
            nn.ReLU(True),
            nn.Conv2d(128, 64, 9),
            nn.ReLU(True),
            nn.Conv2d(64, input_repr_size, 48),
            nn.ReLU(True),
        )
        self.input_recons = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, img_size**2)
        )
        self.input_repr_to_hidden = nn.Linear(input_repr_size, hidden_size)
        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.out_token  = nn.Linear(hidden_size, vocab_size)
    
    def given(self, X):
        self.X = X

    def forward(self, inp):
        x = self.emb(inp)
        c0 = self.get_img_repr()
        c0 = c0.view(1, c0.size(0), c0.size(1))
        c0 = c0.repeat(self.num_layers, 1, 1)
        h0 = c0.clone()
        o, (h, c) = self.lstm(x, (h0, c0))
        o = o.contiguous()
        o = o.view(o.size(0) * o.size(1), o.size(2))
        o = self.out_token(o)
        h = h[-1]
        xrec = self.input_recons(h)
        return o, c0, xrec

    def get_img_repr(self):
        img = self.X
        img = img.view(img.size(0), 1, 64, 64)
        img = self.input_repr(img)
        img = img.view(img.size(0), -1)
        img = self.input_repr_to_hidden(img)
        return img

    def next_token(self, inp, state):
        if self.use_cuda:
            inp = inp.cuda()
        if state is None:
            c0 = self.get_img_repr()
            c0 = c0.view(1, c0.size(0), c0.size(1))
            c0 = c0.repeat(self.num_layers, 1, 1)
            h0 = c0.clone()
            state = h0, c0
        x = self.emb(inp)
        _, state = self.lstm(x, state)
        h, c = state
        h = h[-1] # last layer
        o = self.out_token(h)
        return o, state
