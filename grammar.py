from grammaropt.grammar import build_grammar
rules = r"""
svg = elements
elements = (element "\n" elements) / element
element = atom
atom = path
rect = "<rect " x s y s w s h s st"/>\n"
line = "<line " x1 s y1 s x2 s y2 s st "/>\n"
path = "<path d=\"" "M" s int s int s "C" s int s int "," int s int "," int s int "\"" s fill s stroke s stroke_width "/>\n"
fill = "fill=\"transparent\""
stroke = "stroke=\"black\""
stroke_width = "stroke-width=\"" ("1"/"2"/"3"/"4") "\""
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
int = "0" / "10" / "20" / "30" / "40" / "50" / "60" / "70" / "80" / "90" / "100"
group = "<svg " vb s x s y s w s h ">\n"   element "\n" elements "</svg>\n"
vb = "viewbox=\" 0 0 100 100\""
"""
svg = build_grammar(rules)
