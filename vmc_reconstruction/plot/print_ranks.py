# coding: utf-8
from __future__ import division, print_function
try:
    from itertools import izip_longest as zip_longest
    chr = unichr
except:
    from itertools import zip_longest
from colorama import Style
import sys
from contextlib import contextmanager

bold = lambda s: Style.BRIGHT + s + Style.NORMAL

def indent(strn, depth=1):
    indent_line = lambda l: '\t'*(depth*bool(l)) + l
    return '\n'.join(indent_line(s) for s in strn.split('\n'))

@contextmanager
def IndentBlock(depth=1):
    old_stdout = sys.stdout
    class __IndentStdOut(object):
        def write(self, text):
            old_stdout.write(indent(text, depth))
        def close(self):
            old_stdout.close()
    sys.stdout = __IndentStdOut()
    yield
    sys.stdout = old_stdout

def block(i):
    assert 0 <= i <= 8
    if i == 0: return ""
    base = int("2580", base=16) # start of the "Unicode Block Elements Block"
    return chr(base+i)

def column(val, max, max_height=3):
    ticks = (val*max_height*8)//max
    col = [block(8)]*(ticks//8)
    if ticks%8: col.append(block(ticks%8))
    return col

def join_columns(cols, separator=' '):
    rows = list(zip_longest(*cols, fillvalue=' '))[::-1]
    ret = ""
    for row in rows:
        ret = ret + separator.join(row) + "\n"
    return ret

def prints_ranks(ranks):
    R = max(ranks)
    cols = [column(r,R) for r in ranks]
    return join_columns(cols)

def frame(img):
    lines = img.splitlines()
    for line in lines:
        assert len(line) == len(lines[0])

    ret_lines = []
    ret_lines.append(u"\u250C" + u"\u2500"*len(lines[0]) + u"\u2510")
    for line in lines:
        ret_lines.append(u"\u2502" + line + u"\u2502")
    ret_lines.append(u"\u2514" + u"\u2500"*len(lines[0]) + u"\u2518")
    return "\n".join(ret_lines)

if __name__=="__main__":
    ranks = [15,12,7,4,6,2,1]
    with IndentBlock():
        print(frame(prints_ranks(ranks)))
