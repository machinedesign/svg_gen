import numpy as np

from subprocess import call
from grammar import svg
from grammaropt.random import RandomWalker
from grammaropt.grammar import as_str

from skimage.io import imread, imsave

tpl = """<?xml version="1.0" standalone="no"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{w}\" height=\"{h}\">
{content}
</svg>
"""

def svg_to_array(code, w=100, h=100):
    S = tpl.format(w=w, h=h, content=code)
    with open('out.svg', 'w') as fd:
        fd.write(S)
    call("convert out.svg out.png", shell=True)
    im = imread('out.png')
    return im


if __name__ == '__main__':
    data = np.load('/home/mcherti/work/data/fonts/fonts.npz')
    X = data['X']
    best = None
    best_err = np.inf

    x = 255 - X[0, 0]
    imsave('real.png', x)
    for i in range(10000):
        wl = RandomWalker(svg, min_depth=1, max_depth=3)
        wl.walk()
        s = as_str(wl.terminals)
        sarr = svg_to_array(s, w=64, h=64)
        sarr = 255 - sarr[:, :, 0]
        err = np.abs(x - sarr).mean()
        if err < best_err:
            best_err = err
            best = s
            print('NEW BEST')
            print(best)
            imsave('best.png', sarr)
