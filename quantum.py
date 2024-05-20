from qiskit import QuantumCircuit, Aer, execute, transpile
from qiskit_aer import AerSimulator

from qiskit.circuit import Instruction, Circuit, Qubit, QuantumRegister
from qiskit.circuit.library.standard_gates import CXGate, CCXGate, C3XGate, C4XGate, MCXGate

import matplotlib.pyplot as plt
#%matplotlib inline
from qiskit.visualization import plot_histogram

import os
import sys
import math as m
import numpy as np
import pandas as pd
import sympy

from functools import *
from traceback import format_exc


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





