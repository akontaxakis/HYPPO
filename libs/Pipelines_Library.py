import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.metrics import silhouette_score
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")

def keep_two_digits(number):
    str_number = str(number)
    index_of_decimal = str_number.index('.')
    str_number_no_round = str_number[:index_of_decimal + 2]
    return str_number_no_round




def compute_correlation(data1, data2):
    corr_matrix = np.corrcoef(data1, data2, rowvar=False)
    return np.average(np.abs(np.diag(corr_matrix, k=1)))


def compare_pickles_exact(artifact_dir='artifacts'):
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    equal_pairs = []

    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        with open(file1, 'rb') as f:
            data1 = pickle.load(f)

        for j in range(i + 1, num_files):
            file2 = os.path.join(artifact_dir, files[j])

            with open(file2, 'rb') as f:
                data2 = pickle.load(f)

            if np.array_equal(data1, data2):
                print("found a pair")
                equal_pairs.append((files[i], files[j]))

    return equal_pairs


def compare_pickles(artifact_dir='artifacts', correlation_threshold=0.9):
    files = [f for f in os.listdir(artifact_dir) if f.endswith('.pkl')]
    num_files = len(files)
    highly_correlated_pairs = []
    print(num_files)
    for i in range(num_files):
        file1 = os.path.join(artifact_dir, files[i])

        with open(file1, 'rb') as f:
            data1 = pickle.load(f)

        for j in range(i + 1, num_files):
            file2 = os.path.join(artifact_dir, files[j])

            with open(file2, 'rb') as f:
                data2 = pickle.load(f)

            correlation = compute_correlation(data1, data2)
            print(correlation)
            if correlation >= correlation_threshold:
                highly_correlated_pairs.append((files[i], files[j], correlation))

    return highly_correlated_pairs


def fit_pipeline_with_artifacts(pipeline, X_train, y_train):
    artifacts = {}
    X_temp = X_train.copy()

    for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the classifier step
        X_temp = step_transformer.fit_transform(X_temp, y_train)
        artifacts[step_name] = X_temp.copy()

    # Fit the classifier step
    step_name, step_transformer = pipeline.steps[-1]
    step_transformer.fit(X_temp, y_train)
    artifacts[step_name] = step_transformer

    return artifacts


def create_artifact_graph(artifacts):
    G = nx.DiGraph()

    for i, (step_name, artifact) in enumerate(artifacts.items()):
        G.add_node(step_name, artifact=artifact)
        if i > 0:
            prev_step_name = list(artifacts.keys())[i - 1]
            G.add_edge(prev_step_name, step_name)

    return G


def plot_artifact_graph(G):
    pos = nx.drawing.layout.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
    plt.show()


def get_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps


def generate_pipeline(steps, number_of_steps, task='clustering'):
    pipeline_steps = []
    optional_steps, mandatory_steps = get_steps(steps[:-1])
    steps_count = random.randint(1, len(optional_steps))

    selected_steps = random.sample(optional_steps, steps_count)

    selected_steps = mandatory_steps + selected_steps
    selected_steps.append(steps[number_of_steps - 1])
    for step_name, options in selected_steps:
        selected_option = random.choice(options)
        pipeline_steps.append((step_name, selected_option))
    # print(pipeline_steps)
    return Pipeline(pipeline_steps)


def fit_pipeline_with_artifacts(pipeline, X_train, y_train):
    artifacts = {}
    X_temp = X_train.copy()

    for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the classifier step
        X_temp = step_transformer.fit_transform(X_temp, y_train)
        artifacts[step_name] = X_temp.copy()

    # Fit the classifier step
    step_name, step_transformer = pipeline.steps[-1]
    step_transformer.fit(X_temp, y_train)
    artifacts[step_name] = step_transformer

    return artifacts


def fit_pipeline_with_store_or_load_artifacts(pipeline, X_train, y_train, materialization, artifact_dir='artifacts'):
    os.makedirs(artifact_dir, exist_ok=True)
    artifacts = {}
    X_temp = X_train.copy()
    artifact_name = ""
    for step_name, step_transformer in pipeline.steps[:-1]:  # Exclude the classifier step
        artifact_name = artifact_name + str(step_transformer) + "_";
        artifact_path = os.path.join(artifact_dir, f"{artifact_name}.pkl")

        if os.path.exists(artifact_path):
            with open(artifact_path, 'rb') as f:
                print("load" + artifact_name)
                X_temp = pickle.load(f)
        else:
            X_temp = step_transformer.fit_transform(X_temp, y_train)
            if random.randint(1, 100) < materialization:
                with open(artifact_path, 'wb') as f:
                    pickle.dump(X_temp, f)

        artifacts[step_name] = X_temp.copy()

    # Fit the classifier step
    step_name, step_transformer = pipeline.steps[-1]
    step_transformer.fit(X_temp, y_train)
    artifacts[step_name] = step_transformer

    return artifacts


def compute_pipeline_metrics(artifact_graph, pipeline, uid, X_train, X_test, y_train, y_test, artifacts,mode,
                              scores_dir='metrics',artifact_dir='artifacts',
                             materialization=100):

    os.makedirs(scores_dir, exist_ok=True)
    scores_file = uid + "_scores"

    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")


    if mode == "sampling":
        hs_previous = "sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    step_full_name = hs_previous
    for step_name, step_obj in pipeline.steps:

        step_start_time = time.time()
        if str(step_obj).startswith("GPU"):
            step_full_name = step_full_name + "GPU__" + str(step_obj).removeprefix("GPU_") + "__"
        else:
            step_full_name = step_full_name + str(step_obj) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")

        if hasattr(step_obj, 'fit_transform'):
            X_temp = step_obj.fit_transform(X_temp, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

        elif hasattr(step_obj, 'fit'):
            X_temp = step_obj.fit(X_temp, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

        if hasattr(step_obj, 'predict'):
            #step_obj.predict(X_test)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
        if random.randint(1, 100) < materialization:
            artifacts.append(hs_current)
            with open(artifact_path, 'wb') as f:
                pickle.dump(X_temp, f)

        if hs_previous == "":
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge("source", hs_current, weight=step_time)
            hs_previous = hs_current
        else:
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge(hs_previous, hs_current, weight=step_time)
            hs_previous = hs_current

    end_time = time.time()
    step_start_time = time.time()

    # Check if the pipeline has a classifier
    has_classifier = any(step_name == 'classifier' for step_name, _ in pipeline.steps)

    has_clustering = any(step_name == 'clustering' for step_name, _ in pipeline.steps)

    if has_classifier:
        step_start_time = time.time()
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        end_time = time.time()
        rounded_score = keep_two_digits(score)
        node_name = str(extract_first_two_chars(step_full_name)[-2:]) + "_" + str(rounded_score)
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}","{node_name}","{end_time - step_start_time}"')

    if has_clustering:
        pipeline.fit(X_train, y_train)
        labels = pipeline.predict(X_test)
        score = silhouette_score(X_test, labels)
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}"')


    return artifact_graph, artifacts


def compute_pipeline_metrics_old(artifact_graph, pipeline, uid, X_train, X_test, y_train, y_test,
                                 metrics_dir='metrics'):
    os.makedirs(metrics_dir, exist_ok=True)
    file_name = uid + "_steps_metrics"
    file_name_2 = uid + "_pipelines_score"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")
    scores_path = os.path.join(metrics_dir, f"{file_name_2}.txt")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            print("load" + metrics_path)
            step_times = pickle.load(f)
    else:
        step_times = []
    start_time = time.time()

    step_full_name = ""
    previous = ""

    for step_name, step_obj in pipeline.steps:
        step_start_time = time.time()
        step_full_name = step_full_name + str(step_obj) + "__"
        hs_previous = extract_first_two_chars(previous)
        hs_current = extract_first_two_chars(step_full_name)

        if hasattr(step_obj, 'fit_transform'):
            step_obj.fit_transform(X_train, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append((step_full_name, step_time))

        elif hasattr(step_obj, 'fit'):
            step_obj.fit(X_train, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append((step_full_name, step_time))

        elif hasattr(step_obj, 'predict'):
            step_obj.predict(X_test)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            step_times.append((step_full_name, step_time))

        if previous == "":
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge("source", hs_current, cost=step_time)
            previous = step_full_name
        else:
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge(hs_previous, hs_current, cost=step_time)
            previous = step_full_name

    end_time = time.time()
    step_start_time = time.time()

    # Check if the pipeline has a classifier
    has_classifier = any(step_name == '3.classifier' for step_name, _ in pipeline.steps)

    has_clustering = any(step_name == 'clustering' for step_name, _ in pipeline.steps)

    if has_classifier:
        pipeline.fit(X_train, y_train)
        score = pipeline.score(X_test, y_test)
        step_time = step_end_time - step_start_time
        step_times.append((step_full_name + "score_time", step_time))
        step_times.append((step_full_name + "score", score))
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}"')

    if has_clustering:
        pipeline.fit(X_train, y_train)
        labels = pipeline.predict(X_test)
        score = silhouette_score(X_test, labels)
        step_time = step_end_time - step_start_time
        step_times.append((step_full_name + "score_time", step_time))
        step_times.append((step_full_name + "score", score))
        with open(scores_path, "a") as outfile:
            outfile.write("\n")
            outfile.write(f'"{step_full_name}","{score}"')
    with open(metrics_path, 'wb') as f:
        pickle.dump(step_times, f)
    # print("Pipeline execution time: {}".format(total_time))
    # for step_name, step_time in step_times:
    #     print("Step '{}' execution time: {}".format(step_name, step_time))
    return step_times, artifact_graph


def extract_first_two_chars(s):
    split_strings = s.split('__')
    result = ''.join([substring[:2] for substring in split_strings])
    return result


def print_metrics(metrics_dir='metrics'):
    n_artifacts = 0;
    os.makedirs(metrics_dir, exist_ok=True)
    file_name = "steps_metrics"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")
    with open(metrics_path, 'rb') as f:
        print("load " + metrics_path)
        step_times = pickle.load(f)
    for step_name, step_time in step_times:
        if step_name.endswith(("__store", "__score_time")):
            n_artifacts = n_artifacts + 0
        else:
            n_artifacts = n_artifacts + 1
        print("Step '{}' execution time: {}".format(step_name, step_time))
    print("number of artifacts " + str(n_artifacts))


def compute_loading_times(metrics_dir='metrics', artifacts_dir='artifacts'):
    os.makedirs(metrics_dir, exist_ok=True)
    loading_times = {}
    file_name = "loading_metrics"
    metrics_path = os.path.join(metrics_dir, f"{file_name}.pkl")

    if os.path.exists(metrics_path):
        with open(metrics_path, 'rb') as f:
            # print("load " + metrics_path)
            loading_times = pickle.load(f)
    else:
        loading_times = {}

    files = [f for f in os.listdir(artifacts_dir) if f.endswith('.pkl')]

    for file in files:
        file_path = os.path.join(artifacts_dir, file)

        start_time = time.time()
        with open(file_path, 'rb') as f:
            _ = pickle.load(f)
        f.close()
        loading_time = time.time() - start_time
        if (file in loading_times):
            if (loading_time > loading_times[file]):
                loading_times[file] = loading_time
        else:
            loading_times[file] = loading_time
    # print(len(loading_times))
    with open(metrics_path, 'wb') as f:
        pickle.dump(loading_times, f)

    return loading_times
