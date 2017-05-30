
from __future__ import division # In Python 3.0, // is the floor division operator and / the true division operator. The true_divide(x1, x2) function is equivalent to true division in Python.
import matplotlib.pyplot as plt
import numpy
import math

def colorbar_classes(classes,ax=None,hv="horizontal",colors=None,unique_classes=None,fontsize=8):
    # classes" 0,1,2,...,C-1
    num_samples=len(classes)
    if unique_classes is None:
        unique_classes=numpy.unique(classes)
    else:
        unique_classes=numpy.array(unique_classes)
    num_classes=len(unique_classes)
    num_samples_per_class=[numpy.sum(classes==c) for c in range(num_classes)]
    #xranges=[(,num_samples_per_class[c]) for c in unique_classes]
    yrange=(0,1)
    facecolors=colors[0:num_classes]
    xranges=[]
    xticks=[]
    xmin=1
    for c in range(num_classes):
        xwidth=num_samples_per_class[c]
        xranges.append((xmin,xwidth))
        xticks.extend([xmin+math.floor(xwidth/2)])
        xmin=xmin+xwidth
    
    if hv=="horizontal":
        ax.set_ylim(0,1)
        ax.set_xlim(1,num_samples)
        ax.broken_barh(xranges,yrange,facecolors=facecolors)
        ax.set_xticks(xticks)
        ax.set_xticklabels(unique_classes.astype(str),fontsize=fontsize)
        ax.xaxis.tick_top()
        ax.set_yticklabels([])
    elif hv=="vertical":
        ax.set_ylim(1,num_samples)
        ax.set_xlim(0,1)
        for c in range(num_classes):
            xr=[(0,1)]
            yr=xranges[c]
            color=colors[c]
            ax.broken_barh(xr,yr,facecolors=color)
            #ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks(xticks)
            ax.set_yticklabels(unique_classes.astype(str),fontsize=fontsize)
            ax.invert_yaxis()
            ax.yaxis.tick_right()
            

fig, ax = plt.subplots()
classes=numpy.array([0,0,0,1,1,1,1,2,2,2,2,2],dtype=int)
colors=numpy.array(["red","yellow","green","blue"])
colorbar_classes(classes,ax=ax,hv="vertical",colors=colors,unique_classes=["c1","c2","c3"],fontsize=12)

plt.show()
