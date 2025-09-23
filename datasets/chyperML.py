# run_hyperband.py
import argparse, time, json, csv
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Binarizer, MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, GradientBoostingClassifier
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker
from hpbandster.optimizers import HyperBand
from ConfigSpace import ConfigurationSpace, CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import EqualsCondition

# === NEW ===
import socket

parser = argparse.ArgumentParser(description="Run Hyper with TPOT configuration.")
parser.add_argument(
    'dataset_id',
    type=str,
    nargs='?',                   # makes it optional
    default='jannis',              # 🔹 default dataset
    help='The dataset ID argument (default: iris)'
)

parser.add_argument(
    'id',
    type=int,
    nargs='?',                   # makes it optional
    default=7777,                   # 🔹 default experiment ID / seed
    help='The experiment ID / random seed argument (default: 0)'
)
# === NEW ===
args = parser.parse_args()
data_id = args.dataset_id
seed = args.id

# ----- your ConfigSpace -----
from Cost_estimator.AutoML_data_manager.data_manager import DataManager

def get_tpot_configspace_classifiers_for_SMAC4AC():
    cs = ConfigurationSpace()
    classifier = CategoricalHyperparameter('classifier', [
        'sklearn.naive_bayes.GaussianNB',
        'sklearn.naive_bayes.BernoulliNB',
        'sklearn.tree.DecisionTreeClassifier',
        'sklearn.neighbors.KNeighborsClassifier',
        'sklearn.ensemble.ExtraTreesClassifier',
        'sklearn.ensemble.RandomForestClassifier',
        'sklearn.ensemble.GradientBoostingClassifier'
    ])
    preprocessor = CategoricalHyperparameter('preprocessor', [
        'sklearn.preprocessing.Binarizer',
        'sklearn.preprocessing.MaxAbsScaler',
        'sklearn.preprocessing.MinMaxScaler',
        'sklearn.preprocessing.Normalizer',
        'sklearn.decomposition.PCA',
        'sklearn.preprocessing.StandardScaler'
    ])
    cs.add_hyperparameters([classifier, preprocessor])

    bernoulli_nb_alpha = UniformFloatHyperparameter('BernoulliNB__alpha', 1e-3, 100, log=True)
    bernoulli_nb_fit_prior = CategoricalHyperparameter('BernoulliNB__fit_prior', [True, False])
    cs.add_hyperparameters([bernoulli_nb_alpha, bernoulli_nb_fit_prior])
    cs.add_condition(EqualsCondition(bernoulli_nb_alpha, classifier, 'sklearn.naive_bayes.BernoulliNB'))
    cs.add_condition(EqualsCondition(bernoulli_nb_fit_prior, classifier, 'sklearn.naive_bayes.BernoulliNB'))

    decision_tree_criterion = CategoricalHyperparameter('DecisionTreeClassifier__criterion', ['gini', 'entropy'])
    decision_tree_max_depth = UniformIntegerHyperparameter('DecisionTreeClassifier__max_depth', 1, 10)
    decision_tree_min_samples_split = UniformIntegerHyperparameter('DecisionTreeClassifier__min_samples_split', 2, 20)
    decision_tree_min_samples_leaf = UniformIntegerHyperparameter('DecisionTreeClassifier__min_samples_leaf', 1, 20)
    cs.add_hyperparameters([decision_tree_criterion, decision_tree_max_depth, decision_tree_min_samples_split, decision_tree_min_samples_leaf])
    for hp in [decision_tree_criterion, decision_tree_max_depth, decision_tree_min_samples_split, decision_tree_min_samples_leaf]:
        cs.add_condition(EqualsCondition(hp, classifier, 'sklearn.tree.DecisionTreeClassifier'))

    knn_n_neighbors = UniformIntegerHyperparameter('KNeighborsClassifier__n_neighbors', 1, 100)
    knn_weights = CategoricalHyperparameter('KNeighborsClassifier__weights', ['uniform', 'distance'])
    knn_p = CategoricalHyperparameter('KNeighborsClassifier__p', [1, 2])
    cs.add_hyperparameters([knn_n_neighbors, knn_weights, knn_p])
    for hp in [knn_n_neighbors, knn_weights, knn_p]:
        cs.add_condition(EqualsCondition(hp, classifier, 'sklearn.neighbors.KNeighborsClassifier'))

    et_max_features = UniformFloatHyperparameter('ExtraTreesClassifier__max_features', 0.05, 1.0)
    et_min_split = UniformIntegerHyperparameter('ExtraTreesClassifier__min_samples_split', 2, 20)
    et_min_leaf = UniformIntegerHyperparameter('ExtraTreesClassifier__min_samples_leaf', 1, 20)
    et_criterion = CategoricalHyperparameter('ExtraTreesClassifier__criterion', ['gini', 'entropy'])
    et_bootstrap = CategoricalHyperparameter('ExtraTreesClassifier__bootstrap', [True, False])
    cs.add_hyperparameters([et_max_features, et_min_split, et_min_leaf, et_criterion, et_bootstrap])
    for hp in [et_max_features, et_min_split, et_min_leaf, et_criterion, et_bootstrap]:
        cs.add_condition(EqualsCondition(hp, classifier, 'sklearn.ensemble.ExtraTreesClassifier'))

    rf_max_features = UniformFloatHyperparameter('RandomForestClassifier__max_features', 0.05, 1.0)
    rf_min_split = UniformIntegerHyperparameter('RandomForestClassifier__min_samples_split', 2, 20)
    rf_min_leaf = UniformIntegerHyperparameter('RandomForestClassifier__min_samples_leaf', 1, 20)
    rf_criterion = CategoricalHyperparameter('RandomForestClassifier__criterion', ['gini', 'entropy'])
    rf_bootstrap = CategoricalHyperparameter('RandomForestClassifier__bootstrap', [True, False])
    cs.add_hyperparameters([rf_max_features, rf_min_split, rf_min_leaf, rf_criterion, rf_bootstrap])
    for hp in [rf_max_features, rf_min_split, rf_min_leaf, rf_criterion, rf_bootstrap]:
        cs.add_condition(EqualsCondition(hp, classifier, 'sklearn.ensemble.RandomForestClassifier'))

    gb_lr = UniformFloatHyperparameter('GradientBoostingClassifier__learning_rate', 1e-3, 1.0, log=True)
    gb_max_depth = UniformIntegerHyperparameter('GradientBoostingClassifier__max_depth', 1, 10)
    gb_min_split = UniformIntegerHyperparameter('GradientBoostingClassifier__min_samples_split', 2, 20)
    gb_min_leaf = UniformIntegerHyperparameter('GradientBoostingClassifier__min_samples_leaf', 1, 20)
    gb_subsample = UniformFloatHyperparameter('GradientBoostingClassifier__subsample', 0.05, 1.0)
    gb_max_features = UniformFloatHyperparameter('GradientBoostingClassifier__max_features', 0.05, 1.0)
    cs.add_hyperparameters([gb_lr, gb_max_depth, gb_min_split, gb_min_leaf, gb_subsample, gb_max_features])
    for hp in [gb_lr, gb_max_depth, gb_min_split, gb_min_leaf, gb_subsample, gb_max_features]:
        cs.add_condition(EqualsCondition(hp, classifier, 'sklearn.ensemble.GradientBoostingClassifier'))

    binarizer_threshold = UniformFloatHyperparameter('Binarizer__threshold', 0.0, 1.0)
    cs.add_hyperparameter(binarizer_threshold)
    cs.add_condition(EqualsCondition(binarizer_threshold, preprocessor, 'sklearn.preprocessing.Binarizer'))

    pca_svd_solver = CategoricalHyperparameter('PCA__svd_solver', ['randomized'])
    pca_iterated_power = UniformIntegerHyperparameter('PCA__iterated_power', 1, 10)
    cs.add_hyperparameters([pca_svd_solver, pca_iterated_power])
    for hp in [pca_svd_solver, pca_iterated_power]:
        cs.add_condition(EqualsCondition(hp, preprocessor, 'sklearn.decomposition.PCA'))

    normalizer_norm = CategoricalHyperparameter('Normalizer__norm', ['l1', 'l2', 'max'])
    cs.add_hyperparameter(normalizer_norm)
    cs.add_condition(EqualsCondition(normalizer_norm, preprocessor, 'sklearn.preprocessing.Normalizer'))
    return cs

def _cv_mean_score(pipe, Xb, yb, scoring, cv):
    from sklearn.model_selection import cross_val_score
    import numpy as np
    scores = cross_val_score(pipe, Xb, yb, scoring=scoring, cv=cv, n_jobs=1, error_score='raise')
    return float(np.mean(scores))

from concurrent.futures import ProcessPoolExecutor, TimeoutError

# ----- budget-aware train (fixed n_estimators) -----
# ----- budget-aware train (fixed n_estimators) -----
def train(config, X, y, budget: float = 1.0, seed: int = seed,
          scoring: str = 'balanced_accuracy', cv_splits: int = 2):
    pre_name = config['preprocessor']
    if pre_name == 'sklearn.preprocessing.Binarizer':
        pre = Binarizer(threshold=config.get('Binarizer__threshold', 0.0))
    elif pre_name == 'sklearn.preprocessing.MaxAbsScaler':
        pre = MaxAbsScaler()
    elif pre_name == 'sklearn.preprocessing.MinMaxScaler':
        pre = MinMaxScaler()
    elif pre_name == 'sklearn.preprocessing.Normalizer':
        pre = Normalizer(norm=config.get('Normalizer__norm', 'l2'))
    elif pre_name == 'sklearn.preprocessing.StandardScaler':
        pre = StandardScaler()
    elif pre_name == 'sklearn.decomposition.PCA':
        pre = PCA(svd_solver=config.get('PCA__svd_solver', 'randomized'),
                  iterated_power=config.get('PCA__iterated_power', 1))
    else:
        raise ValueError(f"Unknown preprocessor: {pre_name}")

    cls = config['classifier']
    if cls == 'sklearn.ensemble.RandomForestClassifier':
        clf = RandomForestClassifier(
            n_estimators=100,
            max_features=config['RandomForestClassifier__max_features'],
            min_samples_split=config['RandomForestClassifier__min_samples_split'],
            min_samples_leaf=config['RandomForestClassifier__min_samples_leaf'],
            bootstrap=config['RandomForestClassifier__bootstrap'],
            criterion=config['RandomForestClassifier__criterion'],
            random_state=seed
        )
    elif cls == 'sklearn.ensemble.GradientBoostingClassifier':
        clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=config['GradientBoostingClassifier__learning_rate'],
            max_depth=config['GradientBoostingClassifier__max_depth'],
            min_samples_split=config.get('GradientBoostingClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('GradientBoostingClassifier__min_samples_leaf', 1),
            subsample=config.get('GradientBoostingClassifier__subsample', 1.0),
            max_features=config.get('GradientBoostingClassifier__max_features', None),
            random_state=seed
        )
    elif cls == 'sklearn.ensemble.ExtraTreesClassifier':
        clf = ExtraTreesClassifier(
            n_estimators=100,
            max_features=config['ExtraTreesClassifier__max_features'],
            min_samples_split=config['ExtraTreesClassifier__min_samples_split'],
            min_samples_leaf=config['ExtraTreesClassifier__min_samples_leaf'],
            criterion=config['ExtraTreesClassifier__criterion'],
            bootstrap=config['ExtraTreesClassifier__bootstrap'],
            random_state=seed
        )
    elif cls == 'sklearn.tree.DecisionTreeClassifier':
        clf = DecisionTreeClassifier(
            criterion=config['DecisionTreeClassifier__criterion'],
            max_depth=config['DecisionTreeClassifier__max_depth'],
            min_samples_split=config.get('DecisionTreeClassifier__min_samples_split', 2),
            min_samples_leaf=config.get('DecisionTreeClassifier__min_samples_leaf', 1),
            random_state=seed
        )
    elif cls == 'sklearn.neighbors.KNeighborsClassifier':
        clf = KNeighborsClassifier(
            n_neighbors=config['KNeighborsClassifier__n_neighbors'],
            weights=config['KNeighborsClassifier__weights'],
            p=config['KNeighborsClassifier__p']
        )
    elif cls == 'sklearn.naive_bayes.GaussianNB':
        clf = GaussianNB()
    elif cls == 'sklearn.naive_bayes.BernoulliNB':
        clf = BernoulliNB(
            alpha=config['BernoulliNB__alpha'],
            fit_prior=config['BernoulliNB__fit_prior']
        )
    else:
        raise ValueError(f"Unknown classifier: {cls}")

    pipe = Pipeline([('pre', pre), ('clf', clf)])

    # subsample by budget
    frac = float(np.clip(budget, 0.05, 1.0))
    n = len(y)
    rng = np.random.RandomState(seed)
    m = max(50, int(frac * n))
    idx = rng.permutation(n)[:m]
    Xb, yb = X[idx], y[idx]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_cv_mean_score, pipe, Xb, yb, scoring, cv)
        try:
            perf = fut.result(timeout=300)  # timelimit is in SECONDS
            return 1.0 - perf, {
                'perf': perf,
                'elapsed_sec': time.time() - t0,
                'budget_frac': frac,
                'subsample_n': int(m)
            }
        except TimeoutError:
            # kill worker and report timeout as requested
            fut.cancel()
            return 1.0, {
                'perf': 0.0,  # score 0 on timeout
                'elapsed_sec': time.time() - t0,  # fixed at 300 on timeout
                'budget_frac': frac,
                'subsample_n': int(m),
                'exception': 'TimeoutError'
            }
        except Exception as e:
            return 1.0, {
                'perf': 0.0,
                'elapsed_sec': time.time() - t0,
                'budget_frac': frac,
                'subsample_n': int(m),
                'exception': type(e).__name__,
                'msg': str(e)
            }
# ----- HpBandSter worker -----
class SKWorker(Worker):
    def __init__(self, X, y, **kwargs):
        super().__init__(**kwargs); self.X, self.y = X, y
    def compute(self, config, budget, **kwargs):
        loss, info = train(config, self.X, self.y, budget=budget, seed=seed, scoring='balanced_accuracy', cv_splits=3)
        return {'loss': loss, 'info': info}

def main():
    import Pyro4
    Pyro4.config.SERIALIZER = 'pickle'
    try:
        Pyro4.config.SERIALIZERS_ACCEPTED.add('pickle')
    except AttributeError:
        Pyro4.config.SERIALIZERS_ACCEPTED = set(getattr(Pyro4.config, 'SERIALIZERS_ACCEPTED', [])) | {'pickle'}

    # DataManager setup
    iris = DataManager(data_id, r'datasets', replace_missing=True, verbose=3)
    X = iris.data['X_train']
    y = iris.data['Y_train']

    cs = get_tpot_configspace_classifiers_for_SMAC4AC()
    exp_id = f"HB_{seed}_{data_id}"
    run_id = f"hb_run_{exp_id}"

    # === CHANGED: bind to node IP and let OS pick a free port ===
    host_ip = socket.gethostbyname(socket.gethostname())
    NS = hpns.NameServer(run_id=run_id, host=host_ip, port=0)
    ns_host, ns_port = NS.start()

    # === CHANGED: point worker to this NS and host ===
    w = SKWorker(X, y, host=host_ip, run_id=run_id, nameserver=ns_host, nameserver_port=ns_port)
    w.run(background=True)

    from hpbandster.core.result import json_result_logger

    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            return super().default(obj)

    json._default_encoder = NumpyJSONEncoder()

    # === CHANGED: pass nameserver + port and keep run_id consistent ===
    HB = HyperBand(configspace=cs,
                   run_id=run_id,
                   nameserver=ns_host,
                   nameserver_port=ns_port,
                   min_budget=0.1,
                   max_budget=1.0,
                   eta=3,
                   result_logger=json_result_logger(directory=exp_id, overwrite=True),
                   ping_interval=60)

    try:
        res = HB.run(n_iterations=999999)
    finally:
        HB.shutdown(shutdown_workers=True)
        NS.shutdown()


if __name__ == '__main__':
    main()
