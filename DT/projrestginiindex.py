from numpy.core.fromnumeric import argmax
import pandas
import numpy
import copy

class node:
    def __init__(self , Attribute):
        self.Attribute = Attribute
        self.Child = []
        self.Value = []
        self.Examples = None
        self.Gain = None
    
    def add_child(self, treenode):
        self.Child.append(treenode)  
        
    def add_Value(self, treenode):
        self.Value.append(treenode) 


def GiniIndx(Example ,Attribute):
    Gini = 0

    Vk = numpy.bincount(Example[Attribute]) #counter different values
    P = Vk / len(Example[Attribute])  #P(Vk) 0 & 1

    for Prob in P:
        if Prob > 0:
            Gini = Gini +(Prob**2)#Gini index

    return 1-Gini
def Remainder(Examples , Attribute , ResultIndex):
    Remainder = 0

    DiffAttVal = Examples[Attribute].unique()#different values
    # print(DiffAttVal)
    Vk = []
    for Value in DiffAttVal:
        Vk.append(Examples[Examples[Attribute]==Value])# result examples examples
    
    for i in range(len(Vk)):
        prob = Vk[i].shape[0]/Examples.shape[0] #(pk+nk / p+n)
        Remainder = Remainder + prob*GiniIndx(Vk[i] , ResultIndex)#remainder

    return Remainder
def Gain(Examples , Attribute , ResultIndex):
    Gain = GiniIndx(Examples , ResultIndex) - Remainder(Examples , Attribute , ResultIndex)#information gain

    return Gain



def PluralityValue(ParentExamples , ResultIndex):
    length = ParentExamples.shape[0]
    Result = list(ParentExamples[ResultIndex])
    Value1 = Result[0]
    Sum1 = 0
    Sum2 = 0
    for i in range(length):
        if Result[i] == Value1:
            Sum1 = Sum1 + 1
        else:
            Value2 = Result[i]
            Sum2 = Sum2 + 1 
    if Sum1 > Sum2:
        return node(Attribute=Value1)
    else:
        return node(Attribute=Value2)


def Importance(Attributes , Examples , ResultIndex):
    GainValues = []
    for Attribute in Attributes:
        GainValues.append(Gain(Examples=Examples , Attribute=Attribute , ResultIndex=ResultIndex))

    ImportantAttribute = argmax(GainValues)
    GainValue = max(GainValues)
    return Attributes[ImportantAttribute] , GainValue 


def CheckAllSame(Examples , ResultIndex):
    if Examples.empty:
        return
    
    length = Examples.shape[0]
    Result = list(Examples[ResultIndex])

    for i in range(length):
        if Result[i] != Result[0]:
            return -1

    return Result[0]



def DecisionTreeLearning(Examples , Attribute , ParentExamples , ResultIndex , DiffValues , AllIndex ):
    
    Attributes = Attribute
    CheckVa = CheckAllSame(Examples , ResultIndex)
    if Examples.empty:
        return PluralityValue(ParentExamples , ResultIndex=ResultIndex)
    elif Attributes == []:
        return PluralityValue(ParentExamples , ResultIndex=ResultIndex)
    elif CheckVa != -1:
        return node(Attribute=CheckVa)
    else:
        ChoesnAttribute , GainValue = Importance(Examples=Examples , Attributes=Attributes , ResultIndex=ResultIndex)
        Tree = node(Attribute=ChoesnAttribute)
        Tree.Gain = GainValue
        Tree.Examples = Examples
        AttIndex = AllIndex.index(ChoesnAttribute)

        DiffValue = DiffValues[AttIndex]#Different Value of ChoesnAttribute
        Vks = []
        for Value in DiffValue:
            Tree.add_Value(Value)
            Vks.append(Examples[Examples[ChoesnAttribute] == Value])#result examples examples

        AttributesT = copy.deepcopy(Attributes)
        AttributesT.remove(ChoesnAttribute)

        for Vk in Vks:
            SubTree = DecisionTreeLearning(Examples=Vk , Attribute=AttributesT , ParentExamples=Examples ,ResultIndex=ResultIndex , DiffValues=DiffValues , AllIndex=AllIndex )
            SubTree.Examples = Vk
            Tree.add_child(SubTree)

        for i in range(len(Tree.Child)):
            print('\tResponse:'+str(Tree.Child[i].Attribute) +'\n'+ '\tIG:'+str(Tree.Child[i].Gain) +'\n'+ '\tNumberOfExamples:'+str(Tree.Child[i].Examples.shape[0]))
            print('\t' + str(Tree.Value[i])+'\n')

        print(str(Tree.Attribute) +'\n'+ 'IG:'+str(Tree.Gain) +'\n'+ 'NumberOfExamples:'+str(Tree.Examples.shape[0])+'\n')

        return Tree



def FindValues(Examples , Attributes):
    DiffValues = []
    for Attribute in Attributes:
        DiffValues.append(Examples[Attribute].unique())

    return DiffValues


Examples = pandas.read_csv("sliderest.csv")
Attributes = list(Examples.columns)
ResultIndex = Attributes[len(Attributes)-1]
Attributes.remove(ResultIndex)
DiffValues = FindValues(Examples=Examples , Attributes=Attributes)
print('\n')
DecisionTreeLearning(Examples=Examples , Attribute=Attributes , ParentExamples=Examples , ResultIndex=ResultIndex , DiffValues=DiffValues , AllIndex=Attributes )