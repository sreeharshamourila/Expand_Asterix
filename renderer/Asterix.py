"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import json
import struct
import os
import os.path as osp
import numpy as np
from PIL import Image
import glob
from random import randint
import cv2
#filelist = glob.glob('BengaliBMPConvert/*.bmp')
class AsterixRenderer(object):
  """
  Domain specific renderer definition
  Main purpose is to be able to take a scene
  graph and return its corresponding rendered image
  """
  def __init__(self, config):
    self.config = config
    self._init_data()

  def _init_data(self):
    asset_dir = self.config['attributes']['asset_dir']
    self.data = {}

    #with open(osp.join(asset_dir,'labels'), 'rb') as flbl:
    #  _, size = struct.unpack('>II', flbl.read(8))
    #  lbls = np.fromfile(flbl, dtype=np.int8)

    #with open(osp.join(asset_dir,'Images'), 'rb') as fimg:
    Background=asset_dir+"Train/Images/0.jpg"
    #filelist = glob.glob(Image_dir)
    #print(filelist)
    imgs = np.array([np.array(Image.open(Background))*100])


    return

  def render(self, graphs):
    """
    Render a batch of graphs to their 
    corresponding images
    """
    if not isinstance(graphs, list): 
      graphs = [graphs]
    
    # should use multiprocessing here
    # but then won't be able to use renderer
    # objects from inside another class
    return [self._render(g) for g in graphs]

  def _render(self, graph):
    """
    Render a single graph into its 
    corresponding image
    """
    # vars
    labels = []
    print(graph.nodes)
    asset_dir = self.config['attributes']['asset_dir']
    background=asset_dir+"Train/Images/0.jpg"
    agent=asset_dir+"Samples/agent.jpg"
    demon=asset_dir+"Samples/demon.jpg"
    target=asset_dir+"Samples/target.jpg"
    im=cv2.imread(background)
    #img=np.array(Image.open(background))
    for i in range(1,len(graph.nodes),2):
        print(graph.nodes[i])
        print(graph.nodes[i+1])
        lane=graph.nodes[i]['attr']['loc_y']
        char=graph.nodes[i+1]['cls']
        print(char)
        ch=0
        yx=[0]*4
        x=randint(3,153)
        if(char=='hero'):
            yx=[lane-5,lane+5,x,x+7]
            ch=cv2.imread(agent)
        elif(char=='Demon'):
            yx=[lane-5,lane+5,x,x+6]
            ch=cv2.imread(demon)
        elif(char=='target'):
            yx=[lane-4,lane+5,x,x+6]
            ch=cv2.imread(target)
            print(ch.shape)
        #print(ag.shape)
        #print(dem.shape)
        #print(tar.shape)
        print(ch.shape)
        print(yx)
        im[yx[0]:yx[1],yx[2]:yx[3]]=ch
    #cv2.imshow("im",im)
    #cv2.waitKey(0)
    

    return im, labels

if __name__ == '__main__':
  config = json.load(open('data/generator/config/Asterix.json', 'r'))
  re = Renderer(config)
