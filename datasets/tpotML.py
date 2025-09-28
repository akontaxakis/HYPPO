import argparse
import os
import sys

from tpot import TPOTClassifier

if __name__ == '__main__':
    sys.path.append(os.path.abspath('/home/users/a/k/akontaxa/.local/lib/python3.10/site-packages'))
    from tpot import TPOTClassifier
    from Cost_estimator.AutoML_data_manager.data_manager import DataManager
    #python3 /home/users/a/k/akontaxa/autoML.py ${DATASET_ID} ${ID} ${K} ${N}
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run TPOT with provided ID and Dataset ID.')
    parser.add_argument('dataset_id', type=str, help='The dataset ID argument')
    parser.add_argument('id', type=int, help='The ID argument')
    parser.add_argument('K', type=int, help='The K argument')
    parser.add_argument('N', type=int, help='The N argument')
    parser.add_argument('G', type=int, help='The G argument')
    # Parse the arguments
    args = parser.parse_args()

    # Use parsed arguments
    data_id = args.dataset_id
    id = args.id
    K = args.K
    N = args.N
    G = args.G
    # DataManager setup
    datasets = [data_id]    
    iris = DataManager(datasets[0], r'/home/users/a/k/akontaxa/datasets',
                       replace_missing=True,
                       verbose=3)
    X = iris.data['X_train']
    y = iris.data['Y_train']

    # TPOT setup
    tpot = TPOTClassifier(K=K, N=N, data_id=data_id, population_size=G, generations=10, cv=2, verbosity=3, mutation_rate=0.9,
                          crossover_rate=0.1, n_jobs=1, template='Transformer-Classifier',
                          random_state=id, config_dict='TPOT light', max_eval_time_mins=5)
    tpot.fit(X, y)