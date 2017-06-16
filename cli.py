from clize import run

from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform


from subprocess import call
from grammar import svg
from grammaropt.random import RandomWalker
from grammaropt.grammar import as_str, extract_rules_from_grammar
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker 

from skimage.io import imread, imsave

def _weights_init(m, ih_std=0.08, hh_std=0.08):
    if isinstance(m, nn.LSTM):
        m.weight_ih_l0.data.normal_(0, ih_std)
        m.weight_hh_l0.data.normal_(0, hh_std)
    elif isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)

 
tpl = """<?xml version="1.0" standalone="no"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{w}\" height=\"{h}\">
{content}
</svg>
"""


def svg_to_png(inp, out, w=64, h=64):
    call('rsvg -w {} -h {} {} {}'.format(w, h, inp, out), shell=True)

def save_svg(s, out):
    with open(out, 'w') as fd:
        fd.write(s)

def prior():
    wlrand = RandomWalker(svg, min_depth=1, max_depth=5)
    for i in range(100):
        wlrand.walk()
        s = as_str(wlrand.terminals)
        S = tpl.format(w=64, h=64, content=s)
        save_svg(S, 'svg/{:05d}.svg'.format(i))
        svg_to_png('svg/{:05d}.svg'.format(i), 'png/{:05d}.png'.format(i), w=64, h=64)


def rnn():
    data = np.load('/home/mcherti/work/data/fonts/fonts.npz')
    X = data['X']
    x = 255 - X[0, 0]
    xf = x.flatten()
    imsave('real.png', x)

    rules = extract_rules_from_grammar(svg)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    vocab_size = len(rules)
    emb_size = 128
    hidden_size = 128
    lr = 1e-3
    gamma = 0.9
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        num_layers=2)
    model.apply(partial(_weights_init, ih_std=0.08, hh_std=0.08))
 
    optim = torch.optim.Adam(model.parameters(), lr=lr) 

    rnn = RnnAdapter(model, tok_to_id)
    wl = RnnWalker(grammar=svg, rnn=rnn, min_depth=1, max_depth=5)
 
    R_avg = 0.
    R_max = 0.
    for i in range(10000000):
        model.zero_grad()
        loss = 0.
        for _ in range(32):
            wl.walk()
            s = as_str(wl.terminals)
            sarr = svg_to_array(s, w=64, h=64)
            sarr = 255 - sarr[:, :, 0]
            if np.all(sarr==0):
                R = 0
            else:
                R = float((xf==sarr.flatten()).mean())
            if R > R_max:
                R_max = R
                best = s
                print('NEW BEST')
                print(best, R)
                imsave('best.png', sarr)
            loss += R * wl.compute_loss() / len(wl._decisions)
        loss /= 32.
        R_avg = R_avg * gamma + R * (1 - gamma)
        loss.backward()
        optim.step()

if __name__ == '__main__':
    run([prior, rnn])
    
