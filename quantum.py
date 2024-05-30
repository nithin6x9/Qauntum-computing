from qiskit import QuantumCircuit, execute, transpile
from qiskit_aer import Aer
from qiskit_aer import AerSimulator

from qiskit.circuit import Instruction, Circuit, Qubit, QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, CCXGate, C3XGate, C4XGate, MCXGate

import matplotlib.pyplot as plt

from qiskit.visualization import plot_histogram

import os
import sys
import math as m
import numpy as np
import pandas as pd
import sympy

from functools import *
from traceback import format_exc

#2
def group(list,subListLen):
    if subListLen != 0:
        grouped_list = []

        try:len_list =len(list)
        except TypeError as errordesc:
            if repr(errordesc) == 'TypeError("objec of type \'int\' has no len()")':
                len_list = subListLen+1

        for a in range(0,len_list,int(subListLen)):
            if type(list) == type(0): grouped_list.append('')
            else:grouped_list.append(list[a:a+subListLen])

        if grouped_list == []:grouped_list.append('')

        return grouped_list
    
    else:
        pass
def afmtsd(the_list_orginal,chara,ndigits):
    #afmtsd - Add For Making The Same Digits
    the_list = list(the_list_orginal)
    a = [chara]*(ndigits-len(the_list))+the_list

    if type(the_list_orginal) == type(''): return''.join(a)

    else: 
        return a
    

def dstatial(the_function,the_list):
    '''Do Something To All Things In A List'''
    the_new_list = []
    for tla in the_list:
        the_new_list.append(function(tla))

    return the_new_list

#3

data = [0,1]

nbits = 2
lendata,digits = len(data),nbits*2

numsList = []
ProductCBinFinals = []
'''Create Raw Table list of lists to converted to pandas 
    So that it can be printed ad tables
    
    '''
RawTable = []

for i in range(lendata**digits):
    nums = [data[(i//lendata**d)%lendata]for d in range(digits)[::-1]]
    numsGrouped = group(nums,nbits)

    nums0 = numsGrouped[0]
    nums1 = numsGrouped[1]

    '''Product using classical computation 
    (Conver nums0 and nums1, binary  number representation,to decimel number A and B)'''
    A,B = np.sum(2**np.arange(digits//2)[::-1]*nums0),np.sum(2**np.arrange(digits//2)[::-1]*nums1)
    ProductClassical = A * B

    #Convert the Classical product to its binary

    ProductBin = list(afmtsd(bin(ProductClassical)[2:],'0',digits))

    #Finalize the Productbin

    ProductCBinFinal = dstatial(int,ProductBin)

    numsList += [nums]
    ProductCBinFinals += [ProductCBinFinal]

    RawTable.append([i,A,B,nums0,nums1,ProductClassical,nums,ProductCBinFinal])

    pd.set_option('display.max_colwodyh',None)

    #create panfas datafram based on RawTable

    Table = Table.style_properties(**{'text-align':'center'})

    Table




