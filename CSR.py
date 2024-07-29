#Import dependence
#Causal discovery 
import cdt
from cdt import SETTINGS
SETTINGS.verbose=False
SETTINGS.NJOBS=8
#Update 2023.3.14 Add R support
cdt.SETTINGS.rpath = '/usr/bin/Rscript'
from cdt.independence.graph import FSGNN
from cdt.causality.pairwise import GNN
from cdt.utils.graph import dagify_min_edge
from cdt.independence.graph import FSGNN,FSRegression,HSICLasso,Lasso
import dowhy
from dowhy import CausalModel
#Symbolic regression
from gplearn.genetic import SymbolicRegressor,SymbolicClassifier
from hyperopt import hp, tpe, fmin, Trials
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
#Basic
import graphviz
from matplotlib.pyplot import MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

#Load data
data = pd.read_csv('train.csv')
data = data.dropna()
#print(data)
#Finding the structure of the graph
obj = HSICLasso()
start_time = time.time()
ugraph = obj.predict(data,threshold=0.005)
print("Finding undirected graph")
print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))
options = {
        "node_color": '#8EB2E4',
        "width": 1,
        "node_size":400,
        "font_size":10,
        "edge_cmap": plt.cm.Blues,
        "with_labels": True,
    }
nx.draw_networkx(ugraph, **options) # The plot function allows for quick visualization of the graph.
plt.show()
#List results
graph_structure = pd.DataFrame(list(ugraph.edges(data='weight')))
#model = cdt.causality.graph.GES()
#ograph = model.predict(data, ugraph)
gnn = GNN(nruns=4, train_epochs=4, test_epochs=1, batch_size=5)
ograph = dagify_min_edge(gnn.orient_graph(data, ugraph))
print("Finding directed graph")
print("--- Execution time : %4.4s seconds ---" % (time.time() - start_time))
nx.draw_networkx(ograph, **options) # The plot function allows for quick visualization of the graph.
plt.show()
# List results
pd.DataFrame(list(ograph.edges(data='weight')), columns=['Cause', 'Effect', 'Score'])

#Reduce feature space
C = graph_structure[0].values.tolist()
E = graph_structure[1].values.tolist()
reduced_feature_space = []
for i in range(len(C)):
    if C[i] == 'Label':
        reduced_feature_space.append(E[i])
print('Reduced feature space:',reduced_feature_space)
Y = data['Label'].values
X = []
for feature in reduced_feature_space:
    X.append(data[feature].values.tolist())
X = np.array(X).T
print('Check data:',X.shape,Y.shape)

#define functions 
#updated 2024.02.14 find causal estimation by dowhy
#def get_causal(P,Y):
#    data = pd.DataFrame({
#        'X': P,  # Treatment variable
#        'Y': Y   # Outcome variable
#    })
#    model = CausalModel(
#        data=data,
#        treatment='X',
#        outcome='Y',
#        common_causes=[]
#    )
#    
#    identified_estimand = model.identify_effect()
#    
#    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression", method_params={'control_value': 'mean'})
#    
#    return(float(str(estimate).split(':')[-1]))
def get_causal(A,B):
    Fsgnn = FSGNN(train_epochs=20, test_epochs=5, l1=0.1,lr=0.1,batch_size=4)#Tune according to your sample size
    ugraph = Fsgnn.predict(data[[A,B]], threshold=1*1e-5)
    gnn = GNN(nruns=10, train_epochs=20, test_epochs=5, batch_size=4)
    ograph = dagify_min_edge(gnn.orient_graph(data[[A,B]], ugraph))
    pd.DataFrame(list(ograph.edges(data='weight')), columns=['Cause', 'Effect', 'Score'])
    return(list(ograph.edges(data='weight')))

# Hyperprameter search space
space = {
    'population_size': hp.quniform('population_size', 100, 500, 50),
    'generations': hp.quniform('generations', 1, 5, 1),
    'parsimony_coefficient': hp.quniform('parsimony_coefficient', -8, -2, 1),
}

def objective(params):
    parsimony_coefficient = 10**int(params['parsimony_coefficient'])
    population_size = int(params['population_size'])
    generations = int(params['generations'])
    rss = [12,22,32,42,52]
    scores = []
    for rs in rss:
        reg = SymbolicClassifier(parsimony_coefficient=parsimony_coefficient,population_size=population_size,
                               generations=generations,const_range=(0, 0),
                               p_crossover=0.5, p_subtree_mutation=0.2,
                               p_hoist_mutation=0.1, p_point_mutation=0.2,
                               feature_names=reduced_feature_space,
                               function_set=('add', 'sub', 'mul', 'div'),
                               init_depth=(1, 3),random_state=rs)
        reg.fit(X,Y)
        score = reg.score(X, Y)
        scores.append(score)
    return -np.mean(scores)

for i in range(1):
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50,trials=trials)
    loss = []
    for t in trials.trials:
        loss.append(t['result']['loss'])
    print('Loss: ',loss)
    print('Best parameter: ',best)

#Genetic Progress Symbolic Regressioin (SymbolicClassifier,SymbolicRegressor)
def prefix2infix(expression):
    operators = ['+', '-', '*', '/']
    count = 0
    for char in expression:
        if char in operators:
            count += 1
    expression = list(expression)
    lenth = len(expression)
    for n in range(count):
        for i in range(lenth):
            if expression[i] in operators and expression[i+1] == '(' and expression[i+2] not in operators:
                for j in range(lenth-i):
                    if expression[j+i] == ',':
                        expression[j+i] = expression[i]
                        expression[i] = '$'     
                        break
    return "".join(expression).replace('$','').replace(' ','')[1:-1]

R2 = []
Cscore = []
Functions = []
SRX = []
print('Start symbolic regression')
for rs in range(100):
    est_gp = SymbolicClassifier(parsimony_coefficient=float(best['parsimony_coefficient']),population_size=int(best['population_size']),
                               generations=int(best['generations']),const_range=(0, 0),#reduce generations to reduce complexity, however, more complex, more accurate
                               p_crossover=0.5, p_subtree_mutation=0.2,
                               p_hoist_mutation=0.2, p_point_mutation=0.1,
                               feature_names=reduced_feature_space,
                               function_set=('add', 'sub', 'mul', 'div'),
                               init_depth=(1, 3),
                               random_state=rs)#Update Pareto front search.
    #help(SymbolicRegressor)
    est_gp.fit(X,Y)
    data['SR'] = est_gp._program.execute(X)
    SRX.append(est_gp._program.execute(X))
    fct = est_gp._program
    prefix_expression = str(fct).replace("mul", "*").replace("add", "+").replace("div", "/").replace("sub", "-")
    print(rs,'Function:',prefix2infix(prefix_expression))
    Functions.append(prefix2infix(prefix_expression))
    print('Score:',est_gp.score(X, Y))
    R2.append(est_gp.score(X, Y))
    plt.figure(figsize=(3,3))
    plt.scatter(est_gp._program.execute(X),Y)
    plt.show()
    print('Calculating Causal Score --')
    c = get_causal('SR','Label')
    print('Causal Score:', c)
    try:
        Cscore.append(c[0][2])
    except:
        Cscore.append(0)

print('Top functions: ')
sorted_lists = sorted(zip(R2,Cscore,Functions,SRX), key=lambda x: x[0], reverse=True)
R2,Cscore,Functions,SRX = zip(*sorted_lists)
np.save('R2.npy',R2)
np.save('Functions.npy',Functions)
np.save('SRX.npy',SRX)
np.save('Cscore.npy',Cscore)
for i in range(5):
    print(i+1,R2[i],Cscore[i],Functions[i])