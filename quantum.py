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

#4

ProductCBinFinalsArr = np.array(ProductCBinFinals)
ProductCBinColNum = ProductCBinFinalsArr.shape[1]

termsCompr1 = []
prodinSOPLen = []

qubitNums = digits*2+ProductCBinColNum

RawTable = []

for i in range(ProductCBinColNum):
    ProductArr = ProductCBinFinalsArr[:,i]
    ProductArr1Indices = np.arange(len(ProductArr))[ProductArr == 1]


    numsArr = np.array(numsList)
    ProductArr1Inputs = numsArr[ProductArr1Indices]

    inputNums = ProductArr1Inputs.shape[1]

    ProductRawEq1 = []
    symbols = [sympy.Symbol('q%s' % QubitIndex)for QubitIndex in range(inputNums)]

    for ProductArr1InputsCont in ProductArr1Inputs:
        ProductRawEq1 += [[{symbols[inputIndex]:ProductArr1InputsCount[InputIndex] for InputIndex in range(len(ProductArr1InputsCont))}]]

        ProductRawEq2 = np.array(ProductRawEq1)
        ProductRawEq3 = ProductRawEq2.flatten().tolist()

        SOP = sympy.SOPform(symbols,ProductRawEq2)
        SOPProducts = SOP.atoms(sympy.And)

        SOP_NAND = bool(1)

        for SOPProduct in SOPProducts:
            SOP_NAND &= ~SOPProduct
        
        SOP_NAND = ~SOP_NAND

        termsList = [tuple(str[SOPProduct].split('&'))for SOPProduct in SOPProducts]
        termsCompr1.extend(termsList)
        prodinSOPLen.append(len(termsList))

        RawTable.append([f'q{i+digits}',SOP_NAND])
    
    pd.set_option('display.max_col_width',None)

    Table = pd.DataFrame(RawTable,columns = ['Output Qubits','Boolean/Logical Equations'])

    Table = Table.style_properties(**{'text-align':'left'})

    Table = Table.set_table_styles([dict(selector='th',props=[('text-align','center')])])

    #5
    termsCompr2 = []

    RawTable = []

    for termsCompr1Cont in termsCompr1:
        if termsCompr1Cont not in termsCompr2:
            termsCompr2.append(termsCompr1Cont)

            RawTable.append([termsCompr1Cont])

    pd.set_option('display.max_colwidth',None)

    Table = pd.DataFrame(RawTable,columns = ['Inner Logical Qubits'])

    TableStyle1 = Table.style.set_properties(**{'text-align':'left'})

    Table.loc['Count'] = Table[['Inner Logical Qubits']].count()

    qubitNumsInitial = qubitNums 
    qubitNums += len(termsCompr2)

    print('Number of qubits:',qubitNums)

    '''Table
        Number of qubits:20'''
    
    #6
    termsCompr3 = []
    for termsCompr2Index in range(len(termsCompr2)):
        targetQubit = (termsCompr2Index+qubitNumsInitial,)

        QubitMCXGCommand = ()

        for controlQubit in termsCompr2[termsCompr2Index]:
            controlQubitProfile = controlQubit.split('q')

            if controlQubitProfile[0]:
                controlQubitProfile = int(controlQubitProfile[1])+digits*2

            else:
                controlQubitProfile = int(controlQubitProfile[1])

            QubitMCXGCommand += (controlQubitProfile,)
        termsCompr3.append(QubitMCXGCommand+targetQubit)

        print(termsCompr3)
    #7


