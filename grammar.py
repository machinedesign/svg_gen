from grammaropt.grammar import build_grammar
rules = r"""
svg = elements
elements = (element "\n" elements) / element
element = atom / group
atom = path
rect = "<rect " x s y s w s h s st"/>\n"
line = "<line " x1 s y1 s x2 s y2 s st "/>\n"
path = "<path d=\"" "M" s int s int s "C" s int s int "," int s int "," int s int "\"" s fill s stroke s stroke_width  "/>\n"
fill = "fill=\"transparent\""
stroke = "stroke=\"black\""
stroke_width = "stroke-width=\"" ("1" / "2" / "3"/ "4" / "5" / "6") "\""
x = "x=" qint
y = "y=" qint
w = "width=" qint
h = "height=" qint
x1 = "x1=" qint
y1 = "y1=" qint
x2 = "x2=" qint
y2 = "y2=" qint
qint = q pint q
pint = int "%"
q = "\""
s = " "
st = "style=\"stroke:rgb(255,0,0);stroke-width:2\""
group = "<svg " vb s x s y s w s h ">\n"   element "\n" elements "</svg>\n"
vb = "viewbox=\" 0 0 100 100\""
int =  "100" / "10" / "15" / "20" / "25" / "30" / "35"/ "40" /  "45" / "50" / "55" / "60" / "65" / "70" /  "75" / "80" /  "85" / "90" / "95" / "0" / "5"
"""
types = {}
svg = build_grammar(rules, types=types)

template = """<?xml version="1.0" standalone="no"?>
<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{w}\" height=\"{h}\">
{content}
</svg>
"""

W_real, H_real = 64, 64
W, H = 64, 64
min_depth = 1
max_depth = 3
