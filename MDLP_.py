# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys 
sys.path.append(r'E:\evan_mime\部门文件\evan_pycode')

from __future__ import division
import pandas as pd
import numpy as np
from Entropy import entropy, cut_point_information_gain
from math import log
from pandas import DataFrame
from copy import copy


class MDLP_Discretizer():
    def __init__(mydat, dep, discol):
        self.mydat = mydat
        self.dep = dep
        self.discol = discol
        
        self._data_new = self.mydat.copy(deep = True)
    
    def 