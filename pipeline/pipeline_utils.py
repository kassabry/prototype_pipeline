# tensorflow backend
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'
# vae stuff
from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
# import scientific py
import numpy as np
import pandas as pd
# rdkit stuff
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import PandasTools
from rdkit import DataStructs
# plotting stuff
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.display import SVG, display
#%config InlineBackend.figure_format = 'retina'
#%matplotlib inline
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_useSVG=False
from rdkit.Chem.PandasTools import FrameToGridImage
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.Draw import MolToImage
import IPython
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import math
import random
from operator import itemgetter
import os
import joblib
from matplotlib.collections import LineCollection
from sklearn.neural_network import MLPClassifier


'''Methods'''

def encode_smiles(pria_smiles, pria_targets, vae):
    encoded_pria_smiles = []
    failed_pria_smiles = []
    encoded_pria_targets = []
    for i in range(len(pria_smiles)):
        try:
            smile_1 = mu.canon_smiles(pria_smiles[i])
            X_1 = vae.smiles_to_hot(smile_1, canonize_smiles=True)
            encoded_pria_smiles.append(vae.encode(X_1))
            encoded_pria_targets.append(pria_targets[i])
        except:
            failed_pria_smiles.append(pria_smiles[i])
            continue
    return encoded_pria_smiles, encoded_pria_targets, failed_pria_smiles

def sample_dataset(dataset, data_targets, budget, seed=1):
    original_df = dataset.copy()
    original_df['Target'] = data_targets

    positive_df = original_df[original_df['Target'] == 1]

    sample_df = original_df.sample(budget, random_state=seed, replace=False)
    if(sample_df['Target'].sum() >= 1):
        return sample_df
    else:
        sample_df = sample_df[:-1]
        sample_df = pd.concat([sample_df, positive_df.sample(1, random_state=seed, replace=True)]) # --->
        ## potentially set ignore_index to true -- might cause issues with indices later though

        return sample_df

''' Make a note to fix the poor optimization of this for loop '''
''' The optimizer holds onto all the points so the runtime will get longer with more iterations '''
def get_next_points(optimizer, budget):
    points = []
    for i, res in enumerate(optimizer.res):
        #print("Iteration {}: \n\t{}".format(i, res))
        single_point = []
        for i in range(len(res['params'])):
            single_point.append(res['params']['dimension ' + str(i)])
        points.append(np.array(single_point).reshape(len(single_point)))

    return points[-budget:] ### Its a good temporary fix for the optimizer holds onto points problem

def distance_calc(points, dataset, data_targets):
    #copy_targets = data_targets.copy()
    #copy_dataset = dataset.copy()

    new_points = []
    new_points_targets = [] ## for getting true targets
    distances = []
    temp_df = pd.DataFrame(points)
    for i in range(len(points)):
        if(len(dataset[((dataset == temp_df.iloc[i]) | (dataset.isnull() & temp_df.iloc[i].isnull())).all(1)]) != 0):
            new_points.append(points[i])
        else:
            ## Try to optimize this distance calculation
            temp_dist = []
            ## Pop off the point after each iteration
            for index, row in dataset.iterrows():
                temp_dist.append((index, np.linalg.norm(temp_df.iloc[i] - row))) ## Euclidean Distance


            sorted_data = sorted(temp_dist,key=itemgetter(1))
            sorted_index = sorted_data[0][0]
            distances.append(sorted_data[0][1])
            #print(type(sorted_index)), print(sorted_index)

            new_points.append(dataset.iloc[sorted_index])
            new_points_targets.append(data_targets.iloc[sorted_index])

            # removing chosen points from to remove duplicated
            #copy_dataset = copy_dataset.drop(sorted_index, axis='index')
            #copy_targets.pop(sorted_index)

    return new_points, new_points_targets, distances

def flatten(xss):
    return [x for xs in xss for x in xs]
        
    
def visualize(optimizer, all_points, all_targets, sample_count, result_path, budget, iter_count, positive_sample, show_img=False):
    explained_var = []
    ## Bar plot for target count by iteration
    all_targets = np.array(all_targets).reshape(np.array(all_targets).shape[0], np.array(all_targets).shape[1])
    sum_by_iteration = []
    for i in range(len(all_targets)):
        sum_by_iteration.append(sum(all_targets[i]))
        
    plt.bar([i for i in range(1,len(sum_by_iteration)+1)], sum_by_iteration)
    plt.xlabel('Iteration')
    plt.ylabel('Active Count')
    plt.savefig(result_path + '/%s_active_count_by_iteration.png' % (str(sample_count)))
    plt.close()
    
    ## Visualize space
    ## ------ make other method for getting points from optimizer -----
    flattened_targets = flatten(all_targets)

    all_points_normalized = (all_points - all_points.mean()) / all_points.std()
    pca = PCA(n_components=2)
    reduced_all_points = pca.fit_transform(all_points_normalized)
    explained_var.append(pca.explained_variance_ratio_) ## Add first explained_var
    reduced_all_points_df = pd.DataFrame(reduced_all_points, columns=['x', 'y']).reset_index(drop=True)

    cmap_gray = mpl.colors.ListedColormap(['lightgray', 'forestgreen'])

    plt.scatter(x=reduced_all_points_df['x'], y=reduced_all_points_df['y'], c=flattened_targets, cmap=cmap_gray, marker='.',
               alpha=1, edgecolors='none')
    # Adding missed postives to the graph
    #if(positive_sample is None):
    positive_sample = positive_sample.iloc[:, :-1]
    missed_pos_normalized = (positive_sample - positive_sample.mean()) / positive_sample.std()
    missed_pos = pca.fit_transform(missed_pos_normalized)
    missed_pos_df = pd.DataFrame(missed_pos, columns=['x', 'y']).reset_index(drop=True)
        
    plt.scatter(x=missed_pos_df['x'], y=missed_pos_df['y'], c='red', marker='x',
                alpha=0.9, edgecolors='none')
        
    plt.savefig(result_path + '/%s_pca_space.png' % (str(sample_count)))
    plt.close()
    
    # Plotting Bayesian optimizer space
    points = []
    optimizer_targets = []
    for i, res in enumerate(optimizer.res):
        single_point = []
        for i in range(len(res['params'])):
            single_point.append(res['params']['dimension ' + str(i)])
        points.append(np.array(single_point).reshape(len(single_point)))
        
        optimizer_targets.append(res['target'])
    
    optimizer_points = pd.DataFrame(points)
    optimizer_points_normalized = (optimizer_points - optimizer_points.mean()) / optimizer_points.std()
    reduced_optimizer_points = pca.fit_transform(optimizer_points_normalized)
    
    explained_var.append(pca.explained_variance_ratio_) ## Append second explained ratio
    
    reduced_optimizer_points_df = pd.DataFrame(reduced_optimizer_points, columns=['x', 'y']).reset_index(drop=True)

    plt.scatter(x=reduced_optimizer_points_df['x'], y=reduced_optimizer_points_df['y'], c=optimizer_targets, 
                cmap='cubehelix_r', marker='.', alpha=0.9, edgecolors='none')
    plt.scatter(x=missed_pos_df['x'], y=missed_pos_df['y'], c='red', marker='x',
                alpha=0.9, edgecolors='none')
    plt.colorbar()
    plt.savefig(result_path + '/%s_pca_bayesian_space.png' % (str(sample_count)))
    plt.close()
    
    #Plotting distribtuion of forest vs actives/inactives
    ''' Figure out how to iterate through these by the budget count and print to a separate file'''
    for iteration in range(iter_count):
        initial_optimizer = optimizer_targets[budget*iteration:budget*(iteration+1)]
        initial_targets = all_targets[budget*iteration:budget*(iteration+1)]
        x=[i for i in range(budget)]
        #print(len(positive_sample), len(rand_val))
        #print(type(initial_targets), print(np.array(initial_targets)), print(initial_targets))
        y=list(zip(initial_optimizer, flatten(initial_targets)))
        lines = []
        for i, j in zip(x,y):
            pair = [(i, j[0]), (i, j[1])]
            lines.append(pair)
        #print(lines)
        #print(initial_optimizer)
        #print(initial_targets)
        linecoll = LineCollection(lines, colors='k')
        fig, ax = plt.subplots()
        ax.plot(x, [i for (i,j) in y], 'bo', markersize = 4)
        ax.plot(x, [j for (i,j) in y], 'rs', markersize = 4)
        ax.add_collection(linecoll)
        plt.savefig(result_path + 'point_differences/%s_point_forest_differences_%s.png' % (str(sample_count), str(iteration)))
        plt.close()
    
    return explained_var
    
def clean_final_points(all_points, all_targets):
    read_in_data = pd.DataFrame()
    read_in_targets = []
    for i in range(len(all_points)):
        read_in_data = pd.concat([read_in_data, pd.DataFrame(all_points[i])], ignore_index=True)
        temp_targets = pd.DataFrame(all_targets[i]).values.tolist()

        temp_targets = [int(l[0]) for l in temp_targets] ## fix typing of list of lists to list of ints
        read_in_targets.append(temp_targets)
        
    return read_in_data, read_in_targets


    
def plot_forest(forest, x_test, y_test, result_path, sample_count):
    
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
    tree.plot_tree(forest.estimators_[0],
                   filled = True);
    plt.savefig(result_path + '/%s_forest.png' % (str(sample_count)))
    plt.close()
    
    ## Precision Recall Curve
    y_pred = 1 - forest.predict_proba(x_test)[:, 0]
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    #create precision recall curve
    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='red')
    #add axis labels to plot
    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    #display plot
    plt.savefig(result_path + '/%s_forest_precision_recall.png' % (str(sample_count)))
    plt.close()

def iterative_bayesian(budget, iteration_count, dataset, data_targets, model='forest_balanced', initial_random_percent=0.7, seed=1):
    
    random.seed(seed)
    
    forest = ''
    if(model == 'forest_balanced'):
        forest = RandomForestClassifier(n_estimators=8000, max_features=math.log(2), 
                                    min_samples_leaf=1 , class_weight='balanced')
    elif(model == 'forest_none'):
        forest = RandomForestClassifier(n_estimators=8000, max_features=math.log(2), 
                                    min_samples_leaf=1)
    elif(model == 'mlp'):
        forest = MLPClassifier()
    
    ## potentially define black box function here <-- nest the function in here
    def black_box_function(**kwargs):
        ## Just return probability of being an active to maximize
        try:
            return forest.predict_proba(np.array(list(kwargs.values())).astype(float).reshape(1,196))[0][1] ## check to see if the forest returns that properly
        except:
            return forest.predict_proba(np.array(list(kwargs.values())).astype(float).reshape(1,196))[0][0]
    
    logger = JSONLogger(path="./logs.log", reset=True) ## potentially change to false if resets after each iteration (don't think it will though)
    

    
    #next_test_set = []
    next_points = []
    next_targets = []
    all_distances = []
    explore_points = int(initial_random_percent * budget)
    exploit_points = budget - int(initial_random_percent * budget)
    
    initial_df =[]
    all_points = []
    all_targets = []
    
    pbounds = {}
    for i in range(196):
        pbounds['dimension ' + str(i)] = (min(dataset.min()), max(dataset.max()))
        
    optimizer = BayesianOptimization(
                    f=black_box_function,
                    pbounds=pbounds,
                    random_state=seed,
                )
        
        
    for i in range(iteration_count):
        print('Iteration ' + str(i))
        if(i == 0):
            sample_df = sample_dataset(dataset, data_targets, budget, seed)
            initial_df = sample_df.copy() ## for record of starting points
            true_targets = sample_df['Target']
            sample_df = sample_df.drop(['Target'], axis=1)
            
            ## train random forest here 
            forest.fit(sample_df, true_targets)
            ## initialize Bayesian Optimizer 

            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize( #Use some percentage of budget for both
                            init_points=explore_points, ## exploration points
                            n_iter=exploit_points, ## exploitation points
                        )
            
            # Remove the sample_df from the total dataset 
            reduced_dataset = dataset.drop(sample_df.index)
            if(type(data_targets) != pd.core.frame.DataFrame):
                reduced_data_targets = pd.DataFrame(data_targets).drop(sample_df.index)
            else:
                reduced_data_targets = data_targets.drop(sample_df.index)
            
            # reset index to avoid index errors
            reduced_dataset = reduced_dataset.reset_index(drop=True)    
            reduced_data_targets = reduced_data_targets.reset_index(drop=True)
            next_points, next_targets, distances = distance_calc(get_next_points(optimizer, budget), reduced_dataset, reduced_data_targets)
            all_distances.append(distances)
            # For sake of keeping record of each run -- I believe can also be seen through the optimizer.res, but just in case
            ## The optimizer.res will return the ones it found, whereas this should return the ones post-discretization
            #print(len(next_points)), print('---' + str(len(next_targets)))
            all_points.append(next_points)
            all_targets.append(next_targets)
       
            #for i, res in enumerate(optimizer.res):
                #print("Iteration {}: \n\t{}".format(i, res))
        
        else:
            explore_points = max(5, explore_points-1) # slowly lower the ratio of explore and exploit
            exploit_points = min(budget-5, exploit_points+1)
            ## train random forest after initializing points
            forest.fit(pd.DataFrame(next_points), next_targets)
            
            # New optimizer is loaded with previously seen points
            load_logs(optimizer, logs=["./logs.log.json"]); ##### Potentially uncomment if it runs into problems
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            optimizer.maximize( #Use some percentage of budget for both
                            init_points=int(initial_random_percent * budget), ## exploration points
                            n_iter=budget - int(initial_random_percent * budget), ## exploitation points
                        )
            
            reduced_dataset = reduced_dataset.drop(pd.DataFrame(next_points).index)
            if(type(reduced_data_targets) != pd.core.frame.DataFrame):
                reduced_data_targets = pd.DataFrame(reduced_data_targets).drop(pd.DataFrame(next_points).index)
            else:
                reduced_data_targets = reduced_data_targets.drop(pd.DataFrame(next_points).index)
            
            # reset indices to reduce chance of index errors
            reduced_dataset = reduced_dataset.reset_index(drop=True)    
            reduced_data_targets = reduced_data_targets.reset_index(drop=True)
            
            next_points, next_targets = [], []
            next_points, next_targets, distances = distance_calc(get_next_points(optimizer, budget), reduced_dataset, reduced_data_targets)
            all_distances.append(distances)
            # For sake of keeping record of each run -- I believe can also be seen through the optimizer.res, but just in case
            ## The optimizer.res will return the ones it found, whereas this should return the ones post-discretization
            '''----------------- There is a memory leak somewhere and it might be with next_points --------------'''
            '''----- All targets has been adding each run onto each other which is why it has been taking so long ----'''
            #print(len(next_points)), print('---' + str(len(next_targets)))
            all_points.append(next_points)
            all_targets.append(next_targets)

    return optimizer, all_points, all_targets, initial_df, all_distances, forest
