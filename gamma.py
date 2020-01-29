import soundfile as sf
import numpy as np
from math import *
sam = 44100
dur = 2
xs = np.arange(0.,float(dur),1/sam)
notes = [261.63,293.66,329.63,349.23,392.00,440.00,493.88,523.25]
full_cord = [notes[0],notes[2],notes[4],notes[6]]
f = lambda x: sin(x*2*pi*full_cord[int(round(x//0.5))])
c = lambda x: sum([sin(x*-2*pi*w) for w in full_cord])
out_l = [f(t) for t in xs]
out_r = [f(t) for t in np.flip(xs)]
out = [[out_l[r],out_r[r]] for r in range(len(out_l))]
sf.write('rec.wav',out,sam)
 
