from clize import run
import os
import uuid

from functools import partial
import numpy as np

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform
from torch.autograd import Variable

from subprocess import call
from grammar import svg
from grammaropt.random import RandomWalker
from grammaropt.grammar import as_str, extract_rules_from_grammar
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker 


from machinedesign.viz import grid_of_images_default
from skimage.io import imread, imsave
from skimage.transform import resize

from rnn import Model

W_real, H_real = 64, 64
W, H = 64, 64
min_depth = 1
max_depth = 5

tpl = """<?xml version="1.0" standalone="no"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{w}\" height=\"{h}\">
{content}
</svg>
"""

def _weights_init(m, ih_std=0.08, hh_std=0.08):
    if isinstance(m, nn.LSTM):
        m.weight_ih_l0.data.normal_(0, ih_std)
        m.weight_hh_l0.data.normal_(0, hh_std)
    elif isinstance(m, nn.Linear):
        xavier_uniform(m.weight.data)
        m.bias.data.fill_(0)


def save_weights(m):
    if isinstance(m, nn.Linear):
        w = m.weight.data
        if w.size(0)  in (128,) and w.size(1) == W*H:
            w = w.view(w.size(0), 1, W, H)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('feat.png', gr)

 
def svg_to_array(s, w=W, h=H):
    u = '.tmp/' + str(uuid.uuid4())
    S = tpl.format(w=w, h=h, content=s)
    save_svg(S, '{}.svg'.format(u))
    svg_to_png('{}.svg'.format(u), '{}.png'.format(u), w=w, h=h)
    im = imread('{}.png'.format(u))
    os.remove('{}.png'.format(u))
    os.remove('{}.svg'.format(u))
    return im


def svg_to_png(inp, out, w=W, h=H):
    call('convert {} {}'.format(inp, out), shell=True)


def save_svg(s, out):
    with open(out, 'w') as fd:
        fd.write(s)


def from_svg(s):
    sarr = svg_to_array(s, w=W_real, h=H_real)
    if len(sarr.shape) == 3:
        sarr = sarr[:, :, 0]
    sarr = sarr.astype(np.float32)
    sarr /= sarr.max()
    sarr = 1.0 - sarr.astype(np.float32)
    sarr = resize(sarr, (W, H))
    sarr = sarr.astype(np.float32)
    return sarr


def prior():
    wlrand = RandomWalker(svg, min_depth=min_depth, max_depth=max_depth)
    for i in range(100):
        wlrand.walk()
        s = as_str(wlrand.terminals)
        S = tpl.format(w=W, h=H, content=s)
        save_svg(S, 'svg/{:05d}.svg'.format(i))
        x = from_svg(s)
        imsave('png/{:05d}.png'.format(i), x)


def fit():
    rules = extract_rules_from_grammar(svg)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    vocab_size = len(rules)
    emb_size = 128
    hidden_size = 256
    lr = 1e-4
    gamma = 0.9
    model = Model(
        input_size=W*H,
        input_repr_size=128,
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        num_layers=2)
    model.apply(partial(_weights_init, ih_std=0.08, hh_std=0.08))
    rnn = RnnAdapter(model, tok_to_id)
    wl = RnnWalker(grammar=svg, rnn=rnn, min_depth=min_depth, max_depth=max_depth)
    rnd = RandomWalker(svg, min_depth=min_depth, max_depth=max_depth)
    optim = torch.optim.Adam(model.parameters(), lr=lr) 

    slist = []
    for i in range(1000):
        rnd.walk()
        slist.append(as_str(rnd.terminals))

    avg_loss = 0.
    nu = 0
    for i in range(10000):
        for s in slist:
            sarr = from_svg(s)
            X = torch.from_numpy(sarr)
            X = X.view(1, -1)
            model.zero_grad()
            model.X = Variable(X)
            wl = RnnDeterministicWalker.from_str(svg, rnn, s)
            wl.walk()
            loss = wl.compute_loss() / len(wl.decisions)
            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            loss.backward()
            optim.step()
            if nu % 10 == 0:
                print(avg_loss)
                torch.save(model, 'model.th')
                model.apply(save_weights)
            nu += 1
     
 
def rnn():
    data = np.load('fonts.npz')
    x = data['X'][0, 0]
    x = x.astype('float32')
    x /= x.max()
    x = 1.0 - x
    x = resize(x, (W, H))
    x  = x.astype('float32')
    imsave('real.png', x)

    rules = extract_rules_from_grammar(svg)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    vocab_size = len(rules)
    emb_size = 128
    hidden_size = 128
    lr = 1e-3
    gamma = 0.9
    """
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        nb_features=2,
        num_layers=2)
    model.apply(partial(_weights_init, ih_std=0.08, hh_std=0.08))
    """
    model = torch.load('model.th')
    X = torch.from_numpy(x)
    X = X.view(1, -1)
    model.X = Variable(X)

    optim = torch.optim.Adam(model.parameters(), lr=lr) 

    rnn = RnnAdapter(model, tok_to_id)
    wl = RnnWalker(grammar=svg, rnn=rnn, min_depth=min_depth, max_depth=max_depth)
    R_avg = 0.
    R_max = 0.
    for i in range(10000):
        wl.walk()
        s = as_str(wl.terminals)
        sarr = from_svg(s)
        if np.all(sarr==0):
            R = 0
        else:
            R = float((x==sarr).mean())
        R_avg = R_avg * gamma + R * (1 - gamma)
        if R > R_max:
            R_max = R
            best = s
            print('NEW BEST at iter {}'.format(i))
            print(best, R)
            imsave('best.png', sarr)
        model.zero_grad()
        loss = (R - R_avg) * wl.compute_loss() / len(wl._decisions)
        loss.backward()
        optim.step()

if __name__ == '__main__':
    run([prior, rnn, fit])
