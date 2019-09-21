from DecisionTree import *
import pandas as pd
from sklearn.model_selection import *

import numpy as np
import matplotlib.pyplot as plt
import copy
import random


#header=['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class'])
#header = ['buying','maint','doors','persons','lug_boot','safety','class']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data', header=None, names=['buying','maint','doors','persons','lug_boot','safety','class'])
header = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class']
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data', header=None, names=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','class'],nrows=2561)
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)
print("********** Leaf nodes ****************")
leaves = getLeafNodes(t,[])
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t,[])
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))



test = lst[0]
lab = classify(test, t)
test = lst[0:10]
print("Accuracy = " + str(computeAccuracy(test, t)))


trainDF, testDF = train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()
print(train)


train_tree = build_tree(train, header)
acc = computeAccuracy(test, train_tree)
threshold = acc
print("Accuracy on test = " + str(acc))


pruningLabelsDict = {'1': 'Prune nodes one level above leaf', '2': 'Randomly select n nodes and prune', '3': 'Prune nodes sequentially till there\'s a decrease in accuracy'} 

def selectNodesToPrune():
	pruningStratergy = {
		'1' : pruneNodesAboveLeaf,
		'2' : pruneRandomNodes,
        '3' : pruneNodesSequentially
	}
	return pruningStratergy

#Function returns node(ID's) to pruned for pruning strategy - Prune nodes one level above leaf
def pruneNodesAboveLeaf() :
	idOfNodesAboveLeaf = []
	leaves = getLeafNodes(train_tree, [])
	#If left/right child of root is a leaf node, do not prune the root!! Skip the root!
	for leaf in leaves:
		if (leaf.id!=1 and leaf.id!=2):
			if(leaf.id%2):
				#parentID when the child is left-child
				parentId = (leaf.id-1)/2
			else:
				#parentID when the child is right-child
				parentId = (leaf.id-2)/2
			#prevent duplicate id of parent
			if parentId not in idOfNodesAboveLeaf:  
				idOfNodesAboveLeaf.append(parentId)
		else:
			#if lead id is 1/2 skip
			continue;
	return idOfNodesAboveLeaf

#Function returns node(ID's) to pruned for pruning strategy - Randomly select n nodes and prune
def pruneRandomNodes():
	innernodeIdList = []
	totalNodes = 0;
	inneNodes = getInnerNodes(train_tree, [])
	for inner in innerNodes:
		innernodeIdList.append(inner.id)
	random.shuffle(innernodeIdList)
	#Take 1/10th of the total internal node ids
	#Change variable 0.1 to prune different fraction of node. For example to prune 20% of internal node, change 0.1 to 0.2.
	numOfNodesToPrune = int(math.ceil(0.1*(len(innernodeIdList))))
	listOfPruneNodeId = innernodeIdList[:numOfNodesToPrune] 
	#Do not prune node with IDs 0, 1, 2
	for ele in [0,1,2]:
		if ele in listOfPruneNodeId:
			listOfPruneNodeId.remove(ele)
			print('Selected node was : ID ' + str(ele) + '!! Thus removed it!!')
	return listOfPruneNodeId
    
def pruneNodesSequentially():
    local_t_trained = copy.deepcopy(train_tree)
    innerNodes_trained = getInnerNodes(local_t_trained,[])
    newlist = sorted(innerNodes_trained, key=lambda x: (x.depth,x.id), reverse=True)
    newlist = newlist[:-1]
    for i in newlist:
        local_t_trained = prune_tree(local_t_trained,[i.id])
        print("*************Tree after pruning*******")
        print_tree(local_t_trained)
        acc = computeAccuracy(test, local_t_trained)
        print("Accuracy on test = " + str(acc))
        global threshold
        if(acc < threshold):
            break
        else:
            threshold = acc
        
pruningStrategyDict = selectNodesToPrune()
for strategy in pruningStrategyDict:
    print("******* Pruning Strategy : " + pruningLabelsDict[strategy] + " *******")
    local_t_trained = copy.deepcopy(train_tree)
    if(strategy != '3'):
        t_pruned = prune_tree(local_t_trained, pruningStrategyDict[strategy]())
        print("*************Tree after pruning*******")
        print_tree(t_pruned)
        acc = computeAccuracy(test, t_pruned)
        print("Accuracy on test = " + str(acc))
    else:
        pruneNodesSequentially()