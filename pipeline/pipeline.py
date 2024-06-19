import argparse
import random 
import os
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from pipeline_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_location', action='store', default='pria_data/', dest='data_folder', required=True)
    parser.add_argument('--target_name', action='store', default='Keck_Pria_AS_Retest', dest='target_name', required=False)
    parser.add_argument('--sample_size', action='store', default=10000, dest='sample_size', required=True)
    parser.add_argument('--number_of_seeds', action='store', default=5, dest='number_of_seeds', required=True)
    parser.add_argument('--budget', action='store', default=96, dest='budget', required=True)
    parser.add_argument('--number_of_iterations', action='store', default=20, dest='iter_count', required=True)
    parser.add_argument('--sample_seed', action='store', default=284, dest='sample_seed', type=int, required=False)
    parser.add_argument('--model', action='store', default='forest_balanced', dest='model', required=True)
    '''Possible arguments for model: forest_balanced, forest_none, mlp '''

    given_args = parser.parse_args()
    data_folder = str(given_args.data_folder) ## Unsure if I need to string this
    target_name = given_args.target_name
    sample_size = int(given_args.sample_size)
    seed_count = int(given_args.number_of_seeds)
    budget = int(given_args.budget)
    iter_count = int(given_args.iter_count)
    model = given_args.model
    sample_seed = given_args.sample_seed
    
    #vae = VAEUtils(directory='chemvae/models/zinc_properties/')
    vae = VAEUtils(directory='models/zinc_properties/')
    
    ''' Make all the data and target names command line variables'''
    pria_training = pd.read_csv(data_folder + '/training.csv', header=0)
    pria_smiles = list(pria_training['SMILES'])
    pria_targets = list(pria_training[target_name])

    pria_testing = pd.read_csv(data_folder + '/testing.csv', header=0)
    pria_test_smiles = list(pria_testing['SMILES'])
    pria_test_targets = list(pria_testing[target_name])
    
    encoded_pria_smiles, encoded_pria_targets, failed_pria_smiles = encode_smiles(pria_smiles, pria_targets, vae)
    encoded_pria_smiles_test, encoded_pria_targets_test, failed_pria_smiles_test = encode_smiles(pria_test_smiles, pria_test_targets, vae)

    encoded_pria_smiles = np.array(encoded_pria_smiles).reshape(len(encoded_pria_smiles), 196)
    encoded_pria_smiles_test = np.array(encoded_pria_smiles_test).reshape(len(encoded_pria_smiles_test), 196)

    encoded_pria_df = pd.DataFrame(encoded_pria_smiles)
    encoded_pria_test_df = pd.DataFrame(encoded_pria_smiles_test)

    random_seeds = [random.randint(0,10000) for i in range(int(seed_count))] ### change to command line variable
    #sample_size = 15000 ### change to command line variable
    #budget = 96 ### change to command line variable
    #iter_count = 20 ### change to command line variable

    for i in range(len(random_seeds)):
        print('Run ' + str(i))
        encoded_pria_df_sample = encoded_pria_df.copy()
        encoded_pria_targets_sample = encoded_pria_targets.copy()

        encoded_pria_df_sample = sample_dataset(encoded_pria_df_sample, encoded_pria_targets_sample, sample_size, 
                                                seed=sample_seed) # seed from first run with good results
        encoded_pria_targets_sample = encoded_pria_df_sample['Target']
        positive_df = encoded_pria_df_sample[encoded_pria_df_sample['Target'] == 1]
        positive_df.drop(['Target'], axis=1)
        
        encoded_pria_df_sample = encoded_pria_df_sample.drop(['Target'], axis=1)
        
        optimizer, all_points, all_targets, init_df, all_distances, forest = iterative_bayesian(budget, iter_count, 
                                                                                                encoded_pria_df_sample, 
                                                                                                encoded_pria_targets_sample,
                                                                                                model=model,
                                                                                                seed=random_seeds[i])
        
        '''Make all the directories necessary'''
        result_path = 'results/%s_%s_pria_bayesian_%s_%s/' % (str(sample_size), str(budget), str(random_seeds[i]), str(model)) ## Modify results after test (get rid of test in title)
        result_path_imgs = result_path + 'visuals/'
        result_path_data = result_path + 'data/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(result_path_imgs):
            os.makedirs(result_path_imgs)
        if not os.path.exists(result_path_data):
            os.makedirs(result_path_data)
        if not os.path.exists(result_path_imgs + 'point_differences/'):
            os.makedirs(result_path_imgs + 'point_differences/')

        ''' Save all the data''' 
        #print(np.array(all_points).shape) ### It is saving the dataframes weird -- look into shape of all_points before cleaning
        ### all_targets seem to be fine it is only the all_points
        for j in range(len(all_points)):
            pd.DataFrame(all_points[j]).to_csv(result_path_data + '%s_all_points_iter_%s.csv' % (sample_size, str(j)), index=False)
            pd.DataFrame(all_targets[j]).to_csv(result_path_data + '%s_all_targets_iter_%s.csv' % (sample_size, str(j)), index=False)
        init_df.to_csv(result_path_data + '%s_initial_df.csv' % (sample_size), index=False)
        positive_df.to_csv(result_path_data + '%s_positive_df.csv' % (sample_size), index=False)
        ## Writing distances to dataframe
        pd.DataFrame(flatten(all_distances)).to_csv(result_path_data + '%s_distances_df.csv' % (sample_size), index=False)
        plot_forest(forest, encoded_pria_test_df, encoded_pria_targets_test, result_path_imgs, sample_size)
        all_points, all_targets = clean_final_points(all_points, all_targets)
        explained_var = visualize(optimizer, all_points, all_targets, sample_size, result_path_imgs, budget, iter_count, positive_sample=positive_df)
        
        
        #visualize_optimizer(optimizer, result_path+'%s_optimizer_points.png' % (sample_size), positive_sample=positive_df)
        
        with open(result_path + 'meta_text.txt', 'a+') as f:
            f.write('Random Seed: ' + str(random_seeds[i]) + '\n')
            f.write('------------------------------------------------ \n')
            f.write('Total Positives: ' + str(len(positive_df)) + '\n')
            f.write('Missed Positives: ' + str(len(positive_df) - sum(flatten(all_targets))) + '\n')
            f.write('------------------------------------------------ \n')
            f.write('Explained Variance: ')
            for var in explained_var:
                f.write(str(var)  +'---')
            f.write('\n')
            f.write('------------------------------------------------ \n')
            f.write('Average Distance : ' + str(sum(flatten(all_distances)) / len(flatten(all_distances))) + '\n')
            
            ### Maybe add time it took to execute after implementing that
        print('-------------------------------------------------------')