import os
from functools import partial
from clize import run

import pandas as pd

import numpy as np
from skimage.io import imread
from skimage.io import imsave

import torch
import torch.nn as nn
from torch.autograd import Variable

from grammaropt.random import RandomWalker
from grammaropt.grammar import as_str
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.grammar import Vectorizer
from grammaropt.grammar import NULL_SYMBOL

from machinedesign.viz import grid_of_images_default, horiz_merge

from rnn import Model, Sim
from utils import from_svg
from utils import save_svg
from utils import acc
from utils import weights_init
from utils import save_weights

from grammar import svg
from grammar import template
from grammar import min_depth
from grammar import max_depth
from grammar import W, H


def prior(*, folder='train', nb=1000):
    wlrand = RandomWalker(svg, min_depth=min_depth, max_depth=max_depth)
    exprlist = []
    svglist = []
    imagelist = []
    for i in range(nb):
        wlrand.walk()
        s = as_str(wlrand.terminals)
        with open('data/{}/text/{:05d}.txt'.format(folder, i), 'w') as fd:
            fd.write(s)
        S = template.format(w=W, h=H, content=s)
        save_svg(S, 'data/{}/svg/{:05d}.svg'.format(folder, i))
        x = from_svg(s)
        imsave('data/{}/png/{:05d}.png'.format(folder, i), x)
        exprlist.append(s)
        svglist.append(S)
        imagelist.append(x)
    np.savez('data/{}/data.npz'.format(folder), X=imagelist, E=exprlist, S=svglist)


def fit(*, folder='results', resume=False):
    grammar = svg
    vect = Vectorizer(svg)
    vect._init()
    
    # Hypers
    vocab_size = len(vect.tok_to_id)
    emb_size = 128
    hidden_size = 256
    lr = 0.001
    gamma = 0.9
    input_repr_size = 128
    use_cuda = True
    batch_size = 64
    epochs = 2000

    # Model
    if resume:
        model = torch.load('{}/model.th'.format(folder))
        vect = model.vect
        grammar = model.grammar
    else:
        model = Model(
            input_size=W*H,
            input_repr_size=input_repr_size,
            vocab_size=vocab_size, 
            emb_size=emb_size, 
            hidden_size=hidden_size, 
            num_layers=2,
            use_cuda=use_cuda
        )
        model.vect = vect
        model.grammar = svg
        model.apply(partial(weights_init, ih_std=0.08, hh_std=0.08))

    if use_cuda:
        model = model.cuda()

    rnn = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=grammar, rnn=rnn, min_depth=min_depth, max_depth=max_depth, temperature=0.0)
    optim = torch.optim.Adam(model.parameters(), lr=lr) 

    # Data
    data = np.load('data/train/data.npz')
    E = data['E']
    E_raw = E
    X = 1.0 - data['X']
    X = np.array(X)

    E = vect.transform(E)
    E = [[0] + expr for expr in E]
    E = np.array(E).astype('int32')

    #nb = 10
    #E = E[0:nb]
    #X = X[0:nb]

    # Training
    I = E[:, 0:-1]
    O = E[:, 1:]
    crit = nn.CrossEntropyLoss()
    avg_loss = 0.
    avg_precision = 0.
    avg_loss = 0.
    avg_recons = 0.
    nupdates = 0
    stats = []
    lambda_recons = 1
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
            y, xrec = model(inp)
            loss_recons =  ((x - xrec) ** 2).mean()
            loss = crit(y, out) + lambda_recons * loss_recons
            precision = acc(y, out)
            loss.backward()
            optim.step()

            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            avg_precision = avg_precision * gamma + precision.data[0] * (1 - gamma)
            avg_recons = avg_recons * gamma + loss_recons.data[0] *  (1 - gamma)
            stats.append({'loss': loss.data[0], 'precision': precision.data[0], 'recons': loss_recons.data[0]})
            if nupdates % 10 == 0:
                pd.DataFrame(stats).to_csv(os.path.join(folder, 'stats.csv'))
                print('Epoch : {:05d} Avg loss : {:.6f} Avg Precision : {:.6f} Avg recons : {:.6f}'.format(i, avg_loss, avg_precision, avg_recons))
                for idx in range(min(30, len(X))):
                    x = X[idx:idx+1]
                    trueim = x[0]
                    x = torch.from_numpy(x)
                    x = x.view(x.size(0), -1)
                    x = Variable(x)
                    if use_cuda:
                        x = x.cuda()
                    model.given(x)
                    try:
                        wl.walk()
                    except Exception as ex:
                        print(ex)
                        continue
                    expr = as_str(wl.terminals)
                    im = from_svg(expr)
                    im = 1.0 - im
                    predim = im
                    im = grid_of_images_default(np.array([trueim, predim]), shape=(2, 1))
                    imsave(os.path.join(folder, 'example{}.png'.format(idx)), im)
                    """
                    with open(os.path.join(folder, 'txt', 'true{}.txt'.format(idx)), 'w') as fd:
                        fd.write(E_raw[idx])
                    with open(os.path.join(folder, 'txt', 'pred{}.txt'.format(idx)), 'w') as fd:
                        fd.write(expr)
                    """
            nupdates += 1
        
        torch.save(model, '{}/model.th'.format(folder))
        model.apply(partial(save_weights, folder=folder))

def fit_simulator(*, folder='results', resume=False):
    vect = Vectorizer(svg)
    vect._init()
    
    # Hypers
    vocab_size = len(vect.tok_to_id)
    emb_size = 128
    hidden_size = 256
    lr = 0.001
    gamma = 0.9
    use_cuda = True
    batch_size = 64
    epochs = 2000

    # Model
    if resume:
        model = torch.load('{}/sim.th'.format(folder))
        vect = model.vect
    else:
        model = Sim(
            img_size=W,
            vocab_size=vocab_size, 
            emb_size=emb_size, 
            hidden_size=hidden_size, 
            num_layers=2,
        )
        model.vect = vect
        model.grammar = svg
        model.apply(partial(weights_init, ih_std=0.08, hh_std=0.08))

    if use_cuda:
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    # Data
    data = np.load('data/train/data.npz')
    E = data['E']
    X = 1.0 - data['X']
    X = np.array(X)

    E = vect.transform(E)
    E = [[0] + expr for expr in E]
    E = np.array(E).astype('int32')

    #nb = 10
    #E = E[0:nb]
    #X = X[0:nb]

    # Training
    avg_loss = 0.
    nupdates = 0
    stats = []
    for i in range(epochs):
        for j in range(0, len(E), batch_size):
            inp = E[j:j+batch_size]
            out = X[j:j + batch_size]

            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            
            out = torch.from_numpy(out).float()
            out = out.view(out.size(0), -1)
            out = Variable(out)
            
            if use_cuda:
                inp = inp.cuda()
                out = out.cuda()

            model.zero_grad()
            out_pred = model(inp)
            loss =  ((out - out_pred) ** 2).mean()
            loss.backward()
            optim.step()

            avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
            stats.append({'loss': loss.data[0]})
            if nupdates % 10 == 0:
                pd.DataFrame(stats).to_csv(os.path.join(folder, 'stats_sim.csv'))
                print('Epoch : {:05d} Avg loss : {:.6f}'.format(i, avg_loss))
                pred = out_pred.view(out_pred.size(0), 1, W, H)
                true = out.view(pred.size())
                pred = pred.data.cpu().numpy()
                true = true.data.cpu().numpy()
                true = grid_of_images_default(true)
                pred = grid_of_images_default(pred)
                im = horiz_merge(true, pred)
                imsave('{}/sim.png'.format(folder), im)
            nupdates += 1
        torch.save(model, '{}/sim.th'.format(folder))

def evaluate(*, filename='results/model.th'):
    use_cuda = True
    batch_size = 64

    model = torch.load(filename, map_location=lambda storage, loc: storage)
    if use_cuda:
        model = model.cuda()
    vect = model.vect
    model.use_cuda = use_cuda
    rnn = RnnAdapter(model, tok_to_id=model.vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=model.grammar, rnn=rnn, min_depth=min_depth, max_depth=max_depth, temperature=0.0)
    # Data
    data = np.load('data/test/data.npz')
    E = data['E']
    X = 1.0 - data['X']
    X = np.array(X)

    E = vect.transform(E)
    E = [[0] + expr for expr in E]
    E = np.array(E).astype('int32')
    print(E.shape)
    # Training
    I = E[:, 0:-1]
    O = E[:, 1:]
    precision = []
    reconstruction_error = []
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
        model.given(x)
        y, _ = model(inp)
        _, pred = y.max(1)
        precision.extend((pred == out).float().data.tolist())
        for i in range(x.size(0)):
            model.given(x[i:i+1])
            wl.walk()
            expr = as_str(wl.terminals)
            xrec = from_svg(expr)
            xrec = 1 - xrec
            a = x[i].data.cpu().numpy().reshape(xrec.shape)
            b = xrec
            rec = ((a - b) ** 2).mean()
            im = horiz_merge(a, b)
            imsave('results/eval/{:05d}.png'.format(j+i), im)
            reconstruction_error.append(rec)
        print('Mean precision : {:.5f}, Mean reconstruction error : {:.5f}'.format(np.mean(precision), np.mean(reconstruction_error)))
    

def evaluate_letters():
    model = torch.load('results/model.th', map_location=lambda storage, loc: storage)
    model.use_cuda = False
    rnn = RnnAdapter(model, tok_to_id=model.vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=model.grammar, rnn=rnn, min_depth=min_depth, max_depth=max_depth, temperature=0.0)
    
    for i in range(10):
        x = imread('data/letters/{:05d}.png'.format(i))
        xtrue = x
        x = x[None, :, :]
        x = x.astype('float32')
        x = x / x.max()

        x = torch.from_numpy(x)
        x = x.view(x.size(0), -1)
        x = Variable(x)

        model.given(x)
        wl.walk()
        expr = (as_str(wl.terminals))
        xrec = from_svg(expr)
        xrec = 1 - xrec
        imsave('results/letters/{:05d}-true.png'.format(i), xtrue)
        imsave('results/letters/{:05d}-pred.png'.format(i), xrec)


def search_letters():
    learn = False
    #model = torch.load('results/model.th', map_location=lambda storage, loc: storage)
    #model.use_cuda = False
    model = torch.load('results/model.th')
    model = model.cuda()
    model.use_cuda = True

    rnn = RnnAdapter(model, tok_to_id=model.vect.tok_to_id, begin_tok=NULL_SYMBOL)
    wl = RnnWalker(grammar=model.grammar, rnn=rnn, min_depth=min_depth, max_depth=max_depth+20, temperature=0.0)
    #wl = RandomWalker(grammar=model.grammar, min_depth=min_depth, max_depth=max_depth+20)
    i = 1
    x = imread('data/letters/{:05d}.png'.format(i))
    xtrue = x
    x = x[None, :, :]
    x = x.astype('float32')
    x = x / x.max()
    xtrue = ((x>0.5).astype('float32'))[0]

    x = torch.from_numpy(x)
    x = x.view(x.size(0), -1)
    x = Variable(x)
    if model.use_cuda:
        x = x.cuda()
    model.given(x)
    max_reward = 0
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    for _ in range(1000):
        try:
            wl.walk()
        except Exception as ex:
            print(ex)
            continue
        expr = (as_str(wl.terminals))
        xrec = from_svg(expr)
        xrec = 1 - xrec
        xrec = (xrec > 0.5).astype('float32')
        
        R = float((xrec == xtrue).mean())
        if learn:
            model.zero_grad()
            loss = R * wl.compute_loss()
            loss.backward()
            optim.step()
        print(expr)
        if R > max_reward:
            best = xrec.copy()
            max_reward = R
            imsave('results/best.png', best)
            imsave('results/true.png', xtrue)
        print(R)


def make_letters():
    data = np.load('data/fonts.npz')
    x = data['X'][0:10]
    x = x[:, 0, :, :]
    x = 255 - x
    for i, im in enumerate(x):
        imsave('data/letters/{:05d}.png'.format(i), im)


if __name__ == '__main__':
    run([prior, fit, make_letters, search_letters, evaluate, evaluate_letters, fit_simulator])
