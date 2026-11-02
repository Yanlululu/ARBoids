"""Simultaneous single-pass selection relative to one unchanged source center."""
import numpy as np


def independent_select(center,choices,evaluate,order=None):
    center=np.asarray(center,dtype=np.int64);n=len(center)
    order=list(range(n)) if order is None else list(order)
    if sorted(order)!=list(range(n)):raise ValueError('Each defender must appear once')
    plans=[tuple(center)];by_boat={}
    for i in order:
        by_boat[i]=[]
        for k in sorted(set(map(int,choices[i]))):
            p=center.copy();p[i]=k;key=tuple(p)
            if key not in plans:plans.append(key)
            by_boat[i].append((k,plans.index(key)))
    scores=np.asarray(evaluate(np.asarray(plans)),dtype=float)
    if scores.shape!=(len(plans),) or not np.isfinite(scores).all():raise ValueError('Invalid scorer output')
    selected=center.copy()
    for i in order:selected[i]=min(by_boat[i],key=lambda pair:(scores[pair[1]],pair[0]))[0]
    return selected,{'evaluated_unique':len(plans),'configurations':[list(x) for x in plans],
                     'scores':scores.tolist(),'reference_center':center.tolist(),
                     'selection':'simultaneous_common_source_center'}
