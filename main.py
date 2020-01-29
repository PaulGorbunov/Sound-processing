import sys
import soundfile as sf
import numpy as np
from functools import *
import math
from multiprocessing.dummy import Pool as ThreadPool 
from multiprocessing import Process,Pipe
import time
import pickle
import numpy as np
import sys

f_name = 'dict_1_9.wav'

full = True # time
cords = False # cords in wav
decor_rec = False # recording 
takts = 8 # #of takts
d_r = {0.0:0.0} #change note to

s_time = 0 # start time
e_time = 0 # end time
fi = 1 # speed of music
ctv = [261.626,277.183,293.665,311.127,329.628,349.228,369.994,391.995,415.305,440,466.164,493.883,0.0]
notes = ctv+list(map(lambda x:2*x,ctv))+list(map(lambda x:4*x,ctv))
names = ["do","do#","re","re#","mi","fa","fa#","sol","sol#","la","la#","si","-"]
nlist = {}

data, sam = sf.read(f_name) 
if full:
    s_time = 0 
    e_time =  int(len(data)/sam )
shape = (lambda: 2 if type(data[0]).__name__ in ['ndarray','list'] else 1)() # shape of list in wav
dev = math.ceil(sam/(max(notes))/fi) #/second
dur = e_time - s_time
t1 = sam * s_time
t2 = sam * e_time
ys = data[t1:t2]
if shape == 1:
    funct = lambda x,y: ys[x]#[y]
else:
    funct = lambda x,y: ys[x][y]
    
cou = [0,0]

def begin():
    con = [Pipe() for v in range(shape)]
    proc = [Process(target=main, args=(con[u][1],u)) for u in range(shape)]
    (lambda : [q.start() for q in proc])()
    (lambda : [e.join() for e in proc])()
    res = [t[0].recv() for t in con]
    with open("tmp",'wb') as f:
        pickle.dump(res,f)
    
def main(conn,ind):
    re= [[[u*int(sam/dev)+y*int(sam),(u+1)*int(sam/dev)+y*int(sam)] for u in range(dev)] for y in range(dur)]
    re = reduce(lambda x,y:x+y,re)
    if re[-1][-1] != dur * sam:
        re[-1] = [re[-1][0],dur*sam]
    re = [[n,re[n],ind] for n in range(len(re))]
    cou[1] = len(re)
    pool = ThreadPool(4) 
    out = pool.map(start,re)
    out.sort(key=lambda x:x[0])
    out = [c[1] for c in out]
    conn.send(out)
    
def beauty(au,fl = 1):
    au[0] = [w if len(w) == 1 else [0.0] for w in au[0]]
    au[1] = [w if len(w) == 1 else [0.0] for w in au[1]]
    if fl != 1:
        return 0
    ch = lambda x: x if not 0.0 in x else [0.01]
    au[0][-1] = ch(au[0][-1])
    au[1][-1] = ch(au[1][-1])
    for g in au:
        for k in range(len(g)):
            if 0.0 in g[k]:
                h = k
                while(0.0 in g[h]):
                    h+=1
                g[k] = g[h]
    
def dictant(au):
    if cords:
        beauty(au,0)
    enc = [[nlist[notes.index(t[0])] for t in au[0]]]
    for y in range(len(enc[0])):
        if not '-' in enc[0][y] :
            break
    ind = 0
    for j in range(len(enc)):
        for t in enc[j][:y]:
            enc[ind -1].append(t)
        enc[ind] = enc[j][y:]
        ind +=1
    for h in range(len(enc)):
        inds  = list(filter(lambda x: enc[h][x] == enc[h][x+1],[u for u in range(len(enc[h])-1)]))
        cou = [(u for u in range(1,len(enc[h])+1)) for y in range(len(enc[h]))]
        def rel(cou):
            e = next(cou[0])
            cou.pop(0)
            return e
        enc[h] = [[enc[h][u],rel(cou)] if not u in inds else next(cou[0]) for u in range(len(enc[h]))]
        enc[h] = list(filter(lambda x: type(x).__name__ == 'list',enc[h]))
    #print(enc)
    #t_enc = [[enc[0][t*(int(len(enc[0])/takts)) + s] for s in range(int(len(enc[0])/takts))] for t in range(takts)]
    #print(t_enc)
    #if (t_enc[-1][-1] != enc[0][-1]):
        #t_enc[-1].append(enc[0][-1])
    #enc = t_enc
    print()
    for j in enc:
        print (j)
    
def record(au):
    print ("\non air")
    if cords and decor_rec:
        beauty(au)
    xs = np.arange(0.,float(dur),1/sam)
    out = []
    ps = 0
    de_re = lambda x: d_r[x] if x in d_r.keys() else x
    for z in xs :
        if z >= ((ps+1)/dev):
            ps += 1
        s = [0 for l in range(shape)]
        for j in range(shape):
            for u in au[j][ps]:
                s[j] += math.sin(-2*math.pi*z*de_re(u))
        out.append(s)
    sf.write('new_'+f_name,out,sam)
    
def start(k):
    k1,k2 = k[1]
    ind = k[2]
    res = []
    for u in notes:
        q = sum([funct(t,ind)*math.e**(-2*math.pi*(u/sam)*1j*t) for t in range(k1,k2)])
        q = q / (k2-k1)
        res.append(q)
    res = [math.sqrt((t.real)**2 + (t.imag)**2) for t in res]
    cou[0] +=1
    sys.stdout.write("\r\x1b[K"+"done: "+(str(round((cou[0]/cou[1])*100))+"%").__str__())
    sys.stdout.flush()
    if not cords:
        return (k[0],[notes[res.index(max(res))]])
    mean_val = 1.2*(sum(res)/len(res))
    w = [abs(res[r+1] - res[r]) for r in range(len(res)-1)]
    mean_dis = 1.2*(sum(w)/len(w))
    inds = [ w.index(t) for t in list(filter(lambda x: x>mean_dis,w))]
    if (len(inds)>0):
        inds.append(inds[-1]+1)   
        inds = list(filter(lambda x: res[x]>mean_val,inds))
        ave = 1.2*sum([res[i] for i in inds])/len(inds)
        inds = list(filter(lambda x: res[x] > ave,inds))
        return (k[0],[notes[r] for r in inds])
    return (k[0],[])
    
    
if __name__ == "__main__":
    start_time = time.time()
    for u in range(len(notes)):
        nlist[u] = '('+str(u//len(names))+') '+names[u%len(names)]
    begin()
    with open("tmp","rb") as f:
        inp = pickle.load(f)
    if not cords:
        dictant(inp)
    record(inp)
    print("\n--- %s seconds ---" % (time.time() - start_time))
    
    
