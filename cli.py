import pandas as pd
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
from grammaropt.grammar import Vectorizer
from grammaropt.grammar import NULL_SYMBOL

from machinedesign.viz import grid_of_images_default
from skimage.io import imread, imsave
from skimage.transform import resize

from rnn import Model

W_real, H_real = 64, 64
W, H = 64, 64
min_depth = 1
max_depth = 10

tpl = """<?xml version="1.0" standalone="no"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{w}\" height=\"{h}\">
{content}
</svg>
"""

def acc(pred, true_classes):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true_classes).float().mean()
    return acc


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
        if w.size(1) == W*H:
            w = w.view(w.size(0), 1, W, H)
            gr = grid_of_images_default(np.array(w.tolist()), normalize=True)
            imsave('results/feat.png', gr)

 
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
    call('cairosvg  {} -o {}'.format(inp, out), shell=True)


def save_svg(s, out):
    with open(out, 'w') as fd:
        fd.write(s)


def from_svg(s):
    sarr = svg_to_array(s, w=W_real, h=H_real)
    if len(sarr.shape) == 3:
        sarr = sarr[:, :, 3]
    assert len(sarr.shape) == 2, sarr.shape
    sarr = sarr.astype(np.float32)
    if sarr.max() > 0:
        sarr /= sarr.max()
    sarr = 1.0 - sarr.astype(np.float32)
    sarr = resize(sarr, (W, H), preserve_range=True)
    sarr = sarr.astype(np.float32)
    return sarr


def prior(*, nb=100):
    wlrand = RandomWalker(svg, min_depth=min_depth, max_depth=max_depth)
    exprlist = []
    svglist = []
    imagelist = []
    for i in range(nb):
        wlrand.walk()
        s = as_str(wlrand.terminals)
        with open('data/text/{:05d}.txt'.format(i), 'w') as fd:
            fd.write(s)
        S = tpl.format(w=W, h=H, content=s)
        save_svg(S, 'data/svg/{:05d}.svg'.format(i))
        x = from_svg(s)
        imsave('data/png/{:05d}.png'.format(i), x)
        exprlist.append(s)
        svglist.append(S)
        imagelist.append(x)
    np.savez('data/data.npz', X=imagelist, E=exprlist, S=svglist)


def fit():
    vect = Vectorizer(svg)
    vect._init()
    
    # Hypers
    vocab_size = len(vect.tok_to_id)
    emb_size = 128
    hidden_size = 256
    lr = 0.01
    gamma = 0.9
    input_repr_size = 64
    use_cuda = True
    batch_size = 64
    epochs = 2000

    # Model
    """
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size,
        use_cuda=use_cuda,
    )
    """
    model = Model(
        input_size=W*H,
        input_repr_size=input_repr_size,
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        num_layers=1,
        use_cuda=use_cuda
    )
    #model = torch.load('results/model.th')
    if use_cuda:
        model = model.cuda()

    model.apply(partial(_weights_init, ih_std=0.08, hh_std=0.08))
    rnn = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=svg, rnn=rnn, min_depth=min_depth, max_depth=max_depth, temperature=1.0)
    optim = torch.optim.Adam(model.parameters(), lr=lr) 

    # Data
    data = np.load('data/data.npz')
    E = data['E']
    raw_E = E
    X = 1.0 - data['X']
    E = vect.transform(E)
    E = [[0] + expr for expr in E]
    E = np.array(E).astype('int32')

    # Training
    I = E[:, 0:-1]
    O = E[:, 1:]
    crit = nn.CrossEntropyLoss()
    avg_loss = 0.
    avg_precision = 0.
    avg_loss = 0.
    nupdates = 0
    for i in range(epochs):
        for j in range(0, len(I), batch_size):
            inp = I[j:j+batch_size]
            out = O[j:j+batch_size]
            x = X[j:j + batch_size]

            out = out.flatten()
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            out = torch.from_numpy(out).long()
            out = Variable(out)
            
            x = torch.from_numpy(x)
            x =  x.view(x.size(0), -1)
            x = Variable(x)
            
            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()
                x = x.cuda()

            model.zero_grad()
            model.given(x)
            y = model(inp)
            loss = crit(y, out)
            precision = acc(y, out)
            loss.backward()
            optim.step()

            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            avg_precision = avg_precision * gamma + precision.data[0] * (1 - gamma)
            if nupdates % 10 == 0:
                print('Epoch : {:05d} Avg loss : {:.6f} Avg Precision : {:.6f}'.format(i, avg_loss, avg_precision))

                for idx in range(10):
                    x = X[idx:idx+1]
                    imsave('results/true/true{}.png'.format(idx), x[0])
                    x = torch.from_numpy(x)
                    x = x.view(x.size(0), -1)
                    x = Variable(x)
                    if use_cuda:
                        x = x.cuda()
                    model.given(x)
                    wl.walk()
                    expr = as_str(wl.terminals)
                    im = from_svg(expr)
                    im = 1.0 - im
                    imsave('results/pred/pred{}.png'.format(idx), im)
                    with open('results/txt/pred{}.txt'.format(idx), 'w') as fd:
                        fd.write(expr)
                    with open('results/txt/true{}.txt'.format(idx), 'w') as fd:
                        fd.write(raw_E[idx])
            nupdates += 1
        
        torch.save(model, 'results/model.th')
        model.apply(save_weights)


def rnn():
    x = imread('png/00008.png')
    x = x / x.max()
    x = x > 0.5
    x = x.astype(np.float32)

    imsave('results/real.png', x)
    rules = extract_rules_from_grammar(svg)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    vocab_size = len(rules)
    emb_size = 50
    hidden_size = 30
    lr = 0.1
    gamma = 0.9
    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        nb_features=2,
        num_layers=1)
    model.apply(partial(_weights_init, ih_std=0.08, hh_std=0.08))
    #model = torch.load('model.th')
    
    X = torch.from_numpy(x)
    X = X.view(1, -1)
    model.X = Variable(X)
    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    rnn = RnnAdapter(model, tok_to_id)
    
    #wl = RandomWalker(grammar=svg, min_depth=min_depth, max_depth=max_depth)
    R_avg = 0.
    R_max = 0.

    generated = []
    rewards = []
    for i in range(10000):
        if i == 1000:
            s = open('text/00008.txt').read()
            wl = RnnDeterministicWalker.from_str(svg, rnn, s)
            wl.walk()
        else:
            wl = RnnWalker(grammar=svg, rnn=rnn, min_depth=min_depth, max_depth=max_depth)
            wl.walk()
        s = as_str(wl.terminals)
        y = from_svg(s)
        y = y > 0.5
        y = y.astype(np.float32)
        if (y==0).mean() < 0.001:
            R = -1
        else:
            R = float((y == x).mean())
        R_avg = R_avg * gamma + R * (1 - gamma)
        #print(R, R_avg)
        print(R)        
        generated.append(s)
        rewards.append(R)
        if R > R_max:
            R_max = R
            print('NEW BEST at iter {}'.format(i))
            print(R)
            imsave('results/best-{:05d}.png'.format(i), y)
        #p = np.array(rewards)
        #p = np.exp(p)
        #p /= np.sum(p)
        #idx = np.random.choice(np.arange(len(generated)), p=p)
        #s = generated[idx]
        wl = RnnDeterministicWalker.from_str(svg, rnn, s)
        wl.walk()
        model.zero_grad()
        loss = (R - R_avg) * wl.compute_loss() / len(wl._decisions)
        loss.backward()
        optim.step()
        #print('loss : {}'.format(loss.data[0]))
        if i % 1 == 0:
            imsave('results/{:05d}.png'.format(i), y)
        if i % 10 == 0:
            pd.DataFrame({'rewards': rewards}).to_csv('results/loss.csv')

if __name__ == '__main__':
    run([prior, rnn, fit])
