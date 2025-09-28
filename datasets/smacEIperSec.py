import argparse
import time
from pathlib import Path
import logging
from smac import Scenario, AlgorithmConfigurationFacade
from smac.acquisition.function import EIPS
from smac.initial_design import RandomInitialDesign
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from Cost_estimator.AutoML_data_manager.data_manager import DataManager
from ConfigSpace import Configuration
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter, EqualsCondition

logging.basicConfig(level=logging.INFO)

# Argument parser setup
parser = argparse.ArgumentParser(description="Run SMAC with TPOT configuration.")
parser.add_argument('dataset_id', type=str, help='The dataset ID argument')
parser.add_argument('id', type=int, help='The experiment ID / random seed argument')
parser.add_argument('K', type=int, help='The K argument')
parser.add_argument('N', type=int, help='The N argument')
parser.add_argument('l', type=float, help='The l argument')

# Parse arguments
args = parser.parse_args()
data_id = args.dataset_id
id = args.id
K = args.K
N = args.N
l = args.l
# Load dataset with the provided data_id
iris = DataManager(data_id, r'datasets', replace_missing=True)
X = iris.data['X_train']
y = iris.data['Y_train']


def train_quality_cost(config: Configuration, seed: int = 0) -> tuple[float, float]:
    """Train a model and return both quality (error rate) and cost (runtime)."""
    start_time = time.time()

    # Build pipeline (same as before)
    preprocessor_name = config['preprocessor']
    classifier_name = config['classifier']

    # Define preprocessor
    if preprocessor_name == 'sklearn.preprocessing.Binarizer':
        preprocessor = Binarizer(threshold=config.get('Binarizer__threshold', 0.0))
    elif preprocessor_name == 'sklearn.preprocessing.MaxAbsScaler':
        preprocessor = MaxAbsScaler()
    elif preprocessor_name == 'sklearn.preprocessing.MinMaxScaler':
        preprocessor = MinMaxScaler()
    elif preprocessor_name == 'sklearn.preprocessing.Normalizer':
        preprocessor = Normalizer(norm=config.get('Normalizer__norm', 'l2'))
    elif preprocessor_name == 'sklearn.preprocessing.StandardScaler':
        preprocessor = StandardScaler()
    elif preprocessor_name == 'sklearn.decomposition.PCA':
        preprocessor = PCA(svd_solver=config.get('PCA__svd_solver', 'randomized'),
                           iterated_power=config.get('PCA__iterated_power', 1))
    else:
        raise ValueError(f"Unknown preprocessor: {preprocessor_name}")

    # Define classifier
    if classifier_name == 'sklearn.ensemble.RandomForestClassifier':
        model = RandomForestClassifier(
            n_estimators=100,
            max_features=config['RandomForestClassifier__max_features'],
            min_samples_split=config['RandomForestClassifier__min_samples_split'],
            min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
            bootstrap=config['RandomForestClassifier__bootstrap'],
            criterion=config['RandomForestClassifier__criterion'],
            random_state=seed
        )
    elif classifier_name == 'sklearn.ensemble.GradientBoostingClassifier':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=config['GradientBoostingClassifier__learning_rate'],
            max_depth=config['GradientBoostingClassifier__max_depth'],
            min_samples_split=config.get('GradientBoostingClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('GradientBoostingClassifier__min_samples_leaf', 1),
            subsample=config.get('GradientBoostingClassifier__subsample', 1.0),
            max_features=config.get('GradientBoostingClassifier__max_features', None),
            random_state=seed
        )
    elif classifier_name == 'sklearn.ensemble.ExtraTreesClassifier':
        model = ExtraTreesClassifier(
            n_estimators=100,
            max_features=config['ExtraTreesClassifier__max_features'],
            min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
            min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
            criterion=config['ExtraTreesClassifier__criterion'],
            bootstrap=config['ExtraTreesClassifier__bootstrap'],
            random_state=seed
        )
    elif classifier_name == 'sklearn.tree.DecisionTreeClassifier':
        model = DecisionTreeClassifier(
            criterion=config['DecisionTreeClassifier__criterion'],
            max_depth=config['DecisionTreeClassifier__max_depth'],
            min_samples_split=config.get('DecisionTreeClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('DecisionTreeClassifier__min_samples_leaf', 1),
            random_state=seed
        )
    elif classifier_name == 'sklearn.neighbors.KNeighborsClassifier':
        model = KNeighborsClassifier(
            n_neighbors=config['KNeighborsClassifier__n_neighbors'],
            weights=config['KNeighborsClassifier__weights'],
            p=config['KNeighborsClassifier__p']
        )
    elif classifier_name == 'sklearn.naive_bayes.GaussianNB':
        model = GaussianNB()
    elif classifier_name == 'sklearn.naive_bayes.BernoulliNB':
        model = BernoulliNB(
            alpha=config['BernoulliNB__alpha'],
            fit_prior=config['BernoulliNB__fit_prior']
        )
    else:
        raise ValueError(f"Unknown classifier: {classifier_name}")

    # Construct the pipeline with the preprocessor and classifier
    steps = [('preprocessor', preprocessor), ('model', model)]
    pipeline = Pipeline(steps)

    # Perform Cross-Validation
    cv = StratifiedKFold(n_splits=2, shuffle=False)
    scores = cross_val_score(pipeline, X, y, cv=cv)

    # Compute quality (validation error)
    quality = 1 - np.mean(scores)

    # Compute cost (runtime)
    runtime = time.time() - start_time

    return {"quality": quality, "cost": runtime}  # Must return a tuple (quality, cost)

def get_tpot_configspace_classifiers_for_SMAC4AC():
    cs = ConfigurationSpace()

    # Define the main classifier hyperparameter with all classifier choices
    classifier = CategoricalHyperparameter(
        'classifier', [
            'sklearn.naive_bayes.GaussianNB',
            'sklearn.naive_bayes.BernoulliNB',
            'sklearn.tree.DecisionTreeClassifier',
            'sklearn.neighbors.KNeighborsClassifier',
            'sklearn.ensemble.ExtraTreesClassifier',
            'sklearn.ensemble.RandomForestClassifier',
            'sklearn.ensemble.GradientBoostingClassifier'
        ]
    )
    cs.add(classifier)

    # Define the main preprocessor hyperparameter with all preprocessor choices
    preprocessor = CategoricalHyperparameter(
        'preprocessor', [
            'sklearn.preprocessing.Binarizer',
            'sklearn.preprocessing.MaxAbsScaler',
            'sklearn.preprocessing.MinMaxScaler',
            'sklearn.preprocessing.Normalizer',
            'sklearn.decomposition.PCA',
            'sklearn.preprocessing.StandardScaler'
        ]
    )
    cs.add(preprocessor)

    # BernoulliNB
    bernoulli_nb_alpha = UniformFloatHyperparameter('BernoulliNB__alpha', lower=1e-3, upper=100, log=True)
    bernoulli_nb_fit_prior = CategoricalHyperparameter('BernoulliNB__fit_prior', [True, False])
    cs.add([bernoulli_nb_alpha, bernoulli_nb_fit_prior])
    cs.add(EqualsCondition(bernoulli_nb_alpha, classifier, 'sklearn.naive_bayes.BernoulliNB'))
    cs.add(EqualsCondition(bernoulli_nb_fit_prior, classifier, 'sklearn.naive_bayes.BernoulliNB'))

    # DecisionTreeClassifier
    decision_tree_criterion = CategoricalHyperparameter('DecisionTreeClassifier__criterion', ['gini', 'entropy'])
    decision_tree_max_depth = UniformIntegerHyperparameter('DecisionTreeClassifier__max_depth', lower=1, upper=10)
    decision_tree_min_samples_split = UniformIntegerHyperparameter('DecisionTreeClassifier__min_samples_split', lower=2,
                                                                   upper=20)
    decision_tree_min_samples_leaf = UniformIntegerHyperparameter('DecisionTreeClassifier__min_samples_leaf', lower=1,
                                                                  upper=20)
    cs.add([
        decision_tree_criterion,
        decision_tree_max_depth,
        decision_tree_min_samples_split,
        decision_tree_min_samples_leaf
    ])
    cs.add(EqualsCondition(decision_tree_criterion, classifier, 'sklearn.tree.DecisionTreeClassifier'))
    cs.add(EqualsCondition(decision_tree_max_depth, classifier, 'sklearn.tree.DecisionTreeClassifier'))
    cs.add(EqualsCondition(decision_tree_min_samples_split, classifier, 'sklearn.tree.DecisionTreeClassifier'))
    cs.add(EqualsCondition(decision_tree_min_samples_leaf, classifier, 'sklearn.tree.DecisionTreeClassifier'))

    # KNeighborsClassifier
    knn_n_neighbors = UniformIntegerHyperparameter('KNeighborsClassifier__n_neighbors', lower=1, upper=100)
    knn_weights = CategoricalHyperparameter('KNeighborsClassifier__weights', ['uniform', 'distance'])
    knn_p = CategoricalHyperparameter('KNeighborsClassifier__p', [1, 2])
    cs.add([knn_n_neighbors, knn_weights, knn_p])
    cs.add(EqualsCondition(knn_n_neighbors, classifier, 'sklearn.neighbors.KNeighborsClassifier'))
    cs.add(EqualsCondition(knn_weights, classifier, 'sklearn.neighbors.KNeighborsClassifier'))
    cs.add(EqualsCondition(knn_p, classifier, 'sklearn.neighbors.KNeighborsClassifier'))

    # ExtraTreesClassifier
    extra_trees_max_features = UniformFloatHyperparameter('ExtraTreesClassifier__max_features', lower=0.05, upper=1.0)
    extra_trees_min_samples_split = UniformIntegerHyperparameter('ExtraTreesClassifier__min_samples_split', lower=2,
                                                                 upper=20)
    extra_trees_min_samples_leaf = UniformIntegerHyperparameter('ExtraTreesClassifier__min_samples_leaf', lower=1,
                                                                upper=20)
    extra_trees_criterion = CategoricalHyperparameter('ExtraTreesClassifier__criterion', ['gini', 'entropy'])
    extra_trees_bootstrap = CategoricalHyperparameter('ExtraTreesClassifier__bootstrap', [True, False])
    cs.add([
        extra_trees_max_features,
        extra_trees_min_samples_split,
        extra_trees_min_samples_leaf,
        extra_trees_criterion,
        extra_trees_bootstrap
    ])
    cs.add(EqualsCondition(extra_trees_max_features, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_min_samples_split, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_min_samples_leaf, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_criterion, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))
    cs.add(EqualsCondition(extra_trees_bootstrap, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))

    # RandomForestClassifier
    random_forest_max_features = UniformFloatHyperparameter('RandomForestClassifier__max_features', lower=0.05,
                                                            upper=1.0)
    random_forest_min_samples_split = UniformIntegerHyperparameter('RandomForestClassifier__min_samples_split', lower=2,
                                                                   upper=20)
    random_forest_min_samples_leaf = UniformIntegerHyperparameter('RandomForestClassifier__min_samples_leaf', lower=1,
                                                                  upper=20)
    random_forest_criterion = CategoricalHyperparameter('RandomForestClassifier__criterion', ['gini', 'entropy'])
    random_forest_bootstrap = CategoricalHyperparameter('RandomForestClassifier__bootstrap', [True, False])
    cs.add([
        random_forest_max_features,
        random_forest_min_samples_split,
        random_forest_min_samples_leaf,
        random_forest_criterion,
        random_forest_bootstrap
    ])
    cs.add(EqualsCondition(random_forest_max_features, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_min_samples_split, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_min_samples_leaf, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_criterion, classifier, 'sklearn.ensemble.RandomForestClassifier'))
    cs.add(EqualsCondition(random_forest_bootstrap, classifier, 'sklearn.ensemble.RandomForestClassifier'))

    # GradientBoostingClassifier
    gradient_boosting_learning_rate = UniformFloatHyperparameter('GradientBoostingClassifier__learning_rate',
                                                                 lower=1e-3, upper=1.0, log=True)
    gradient_boosting_max_depth = UniformIntegerHyperparameter('GradientBoostingClassifier__max_depth', lower=1,
                                                               upper=10)
    gradient_boosting_min_samples_split = UniformIntegerHyperparameter('GradientBoostingClassifier__min_samples_split',
                                                                       lower=2, upper=20)
    gradient_boosting_min_samples_leaf = UniformIntegerHyperparameter('GradientBoostingClassifier__min_samples_leaf',
                                                                      lower=1, upper=20)
    gradient_boosting_subsample = UniformFloatHyperparameter('GradientBoostingClassifier__subsample', lower=0.05,
                                                             upper=1.0)
    gradient_boosting_max_features = UniformFloatHyperparameter('GradientBoostingClassifier__max_features', lower=0.05,
                                                                upper=1.0)
    cs.add([
        gradient_boosting_learning_rate,
        gradient_boosting_max_depth,
        gradient_boosting_min_samples_split,
        gradient_boosting_min_samples_leaf,
        gradient_boosting_subsample,
        gradient_boosting_max_features
    ])
    cs.add(EqualsCondition(gradient_boosting_learning_rate, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_max_depth, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(
        EqualsCondition(gradient_boosting_min_samples_split, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(
        EqualsCondition(gradient_boosting_min_samples_leaf, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_subsample, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))
    cs.add(EqualsCondition(gradient_boosting_max_features, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))

    # Define Preprocessing Hyperparameters
    binarizer_threshold = UniformFloatHyperparameter('Binarizer__threshold', lower=0.0, upper=1.0)
    cs.add(binarizer_threshold)
    cs.add(EqualsCondition(binarizer_threshold, preprocessor, 'sklearn.preprocessing.Binarizer'))

    pca_svd_solver = CategoricalHyperparameter('PCA__svd_solver', ['randomized'])
    pca_iterated_power = UniformIntegerHyperparameter('PCA__iterated_power', lower=1, upper=10)
    cs.add([pca_svd_solver, pca_iterated_power])
    cs.add(EqualsCondition(pca_svd_solver, preprocessor, 'sklearn.decomposition.PCA'))
    cs.add(EqualsCondition(pca_iterated_power, preprocessor, 'sklearn.decomposition.PCA'))

    normalizer_norm = CategoricalHyperparameter('Normalizer__norm', ['l1', 'l2', 'max'])
    cs.add(normalizer_norm)
    cs.add(EqualsCondition(normalizer_norm, preprocessor, 'sklearn.preprocessing.Normalizer'))

    return cs

import yaml

def update_log_filename(file_path: str, new_filename: str) -> str:
    """
    Reads a YAML logging configuration file, updates the filename for the file handler,
    saves it to a different file with a prefixed name 'log_' + new_filename, and returns
    the path to that file.

    Args:
        file_path (str): Path to the YAML logging configuration file.
        new_filename (str): New filename for the file handler.
        output_directory (str, optional): Directory to save the updated configuration file.
            If not specified, the new file is saved in the same directory as file_path.

    Returns:
        str: Path to the saved updated YAML file.
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the filename in the file handler
    if 'handlers' in config and 'file' in config['handlers']:
        config['handlers']['file']['filename'] = new_filename

    # Define the output file path with 'log_' prefix and the new filename
    output_file_path = f"log_{new_filename}"

    # Save the modified configuration to the new file
    with open(output_file_path, 'w') as file:
        yaml.safe_dump(config, file)

    print(f"Log filename updated to '{new_filename}' and saved as '{output_file_path}'.")
    return output_file_path


if __name__ == '__main__':

    from Cost_estimator.AutoML_data_manager.data_manager import DataManager

    # SMAC configuration space and logging
    configspace = get_tpot_configspace_classifiers_for_SMAC4AC()
    path = r"logging_ak.yaml"
    exp_id = f"SMAC_{data_id}_{id}_{K}_{N}_{l}.log"
    path = update_log_filename(path, exp_id)
    logging_path = Path(path)

    # Define the SMAC scenario
    scenario = Scenario(
        configspace=configspace,
        n_trials=10000,
        deterministic=True,
        objectives=["quality", "cost"],
        seed=id,
        data_id=data_id,
        N=N,
        output_directory="smac_output",
        K=K,
        l=l,
        trial_walltime_limit=300
    )

    # Define the initial design for SMAC
    if N==1 or N==41:
        initial_design = RandomInitialDesign(scenario, n_configs=20)
    else:
        initial_design = RandomInitialDesign(scenario, n_configs=40)

    # Configure SMAC
    from smac.model.multi_objective_model import MultiObjectiveModel

    from smac.model.random_forest import RandomForest
    from smac.model.multi_objective_model import MultiObjectiveModel

    # Define two separate RandomForest models: one for quality, one for cost
    rf_quality = RandomForest(configspace=configspace, seed=id)
    rf_cost = RandomForest(configspace=configspace, seed=id)

    # Multi-objective model wrapper
    multi_obj_model = MultiObjectiveModel(models=[rf_quality, rf_cost], objectives=["quality", "cost"], seed=id)

    smac = AlgorithmConfigurationFacade(
        scenario=scenario,
        target_function=train_quality_cost,  # Must return (quality, cost)
        initial_design=initial_design,
        acquisition_function=EIPS(),  # EIPS expects two objectives
        model=multi_obj_model,  # Use the available model
        logging_level=logging_path
    )

    # Run SMAC optimization to find the best configuration
    best_config = smac.optimize()
    print("Best configuration found:", best_config)
