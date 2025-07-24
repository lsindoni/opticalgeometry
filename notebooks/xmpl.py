#!/usr/bin/env python
# coding: utf-8

# In[97]:


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam


# In[329]:


def distance(t1, t2):
    return torch.sqrt(torch.sum((t1 - t2)**2))

def distance2(t1, t2):
    return (torch.sum((t1 - t2)**2))


# In[6]:


_v1, _v2 = torch.tensor([1, 0]), torch.tensor([0,1])


# In[79]:


def embed(r, theta, func):
    return torch.stack([r * torch.cos(theta), r * torch.sin(theta), func(r)]) 


# In[218]:


def f0(r):
    return 0 * r

def f1(r):
    return torch.sqrt(r - 1)


# In[219]:


def initialise(r1, r2, theta2, npts):
    if r1 == r2:
        rs = np.ones(npts) * r1
    else:
        rs = np.linspace(r1, r2, npts)
    if theta2 == 0:
        thetas = np.ones(npts) * 0
    else:
        thetas = np.linspace(0, theta2, npts)
    rs = [torch.tensor(r, requires_grad=True) for r in rs[1:-1]]
    thetas = [torch.tensor(t, requires_grad=True) for t in thetas[1:-1]]
    return rs, thetas


# In[436]:


class Curve:
    def __init__(self, r1, r2, theta2, npts, func):
        self._r1 = r1
        self._r2 = r2
        self._theta2 = theta2
        self._npts = npts
        self.r1 = torch.tensor(r1)
        self.r2 = torch.tensor(r2)
        self.theta1 = torch.tensor(0.0)
        self.theta2 = torch.tensor(theta2)
        rs, thetas = initialise(r1, r2, theta2, npts)
        self.rs = rs
        self.thetas = thetas
        self.func = func
        self.p1 = embed(self.r1, self.theta1, self.func)
        self.p2 = embed(self.r2, self.theta2, self.func)
        self.optim = Adam(rs + thetas, lr=0.01)

    def embed(self):
        pts = []
        for r, t in zip(self.rs, self.thetas):
            pts.append(embed(r, t, self.func))
        return pts

    def length(self):
        pts = self.embed()
        d = distance(self.p1, pts[0])
        #cnt = 1
        for i in range(len(pts) - 1):
            d = d + distance(pts[i], pts[i+1])
            #cnt += 1
        d = d + distance(pts[-1], self.p2)
        #print(cnt)
        return d

    def lengthpen(self, l=0.1):
        pts = self.embed()
        d = distance(self.p1, pts[0])
        v = d**2
        #cnt = 1
        for i in range(len(pts) - 1):
            delta = distance(pts[i], pts[i+1])
            d = d + delta
            v = v + delta**2
            #cnt += 1
        delta = distance(pts[-1], self.p2)
        d = d + delta
        v = v + delta**2
        #print(cnt)
        return d

    def length2(self):
        pts = self.embed()
        d = distance2(self.p1, pts[0])
        #cnt = 1
        for i in range(len(pts) - 1):
            d = d + distance2(pts[i], pts[i+1])
            #cnt += 1
        d = d + distance2(pts[-1], self.p2)
        #print(cnt)
        return d

    def straighten(self, nsteps):
        func = self.lengthpen
        losses = []
        for istep in range(nsteps):
            self.optim.zero_grad()
            output = func()
            output.backward()
            self.optim.step()
            losses.append(output.detach().numpy())
        return losses


# In[463]:


def plot(pts):
    fig = plt.figure()
    #ax = fig.add_subplot(projection='3d')
    ax = fig.add_subplot()
    xs = [p[0].detach() for p in pts]
    ys = [p[1].detach() for p in pts]
    zs = [p[2].detach() for p in pts]
    #ax.scatter(xs, ys, zs, marker=".")
    ax.scatter(xs, ys, marker=".")
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    #ax.set_zlabel('Z Label')
    ax.set_aspect("equal")
    ax.set_xlim([-15, 15])
    ax.set_ylim([-15, 15])

    plt.show()


# In[464]:


PI = 3.1415926


# In[465]:


plt.plot(np.arctan(np.arange(100)/ 10))


# In[496]:


crvs = [Curve(10, np.sqrt(100 + i**2), np.pi - np.arctan(i/10), 51, f1) for i in range(10)]


# In[497]:


crv = Curve(10, 10, PI, 51, f1)


# In[498]:


_ = crv.embed()


# In[499]:


_s = []
for _c in crvs:
    _s += _c.embed()


# In[500]:


plot(_s)


# In[501]:


crv.length()


# In[503]:


for _i, _c in enumerate(crvs):
    print(_i)
    _c.straighten(3000)


# In[ ]:


#plt.plot(_losses)


# In[504]:


_s = []
for _c in crvs:
    _s += _c.embed()


# In[505]:


_pts = crv.embed()


# In[506]:


plot(_s)


# In[ ]:




