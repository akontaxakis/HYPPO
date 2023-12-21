import os
import pickle
import random
import time
import hashlib
import re
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from memory_profiler import memory_usage
from sklearn.metrics import silhouette_score
from imblearn.pipeline import Pipeline
from pympler import asizeof

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


def plot_artifact_graph(G):
    pos = nx.drawing.layout.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10)
    plt.show()


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


def get_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps


def get_all_steps(steps):
    mandatory_steps = []
    optional_steps = []
    for step_name, options in steps:
        if (str(step_name)[0].isdigit()):
            optional_steps.append((step_name, options))
        else:
            mandatory_steps.append((step_name, options))
    return optional_steps, mandatory_steps


def generate_pipeline(steps, number_of_steps, task='random_no'):
    pipeline_steps = []
    optional_steps, mandatory_steps = get_steps(steps[:-1])
    if task == "random":
        steps_count = random.randint(1, len(optional_steps))
        selected_steps = random.sample(optional_steps, steps_count)
        selected_steps = mandatory_steps + selected_steps
    else:
        selected_steps = mandatory_steps

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


def compute_pipeline_metrics(artifact_graph, pipeline, uid, X_train, X_test, y_train, y_test, artifacts, mode,
                             scores_dir='metrics', artifact_dir='artifacts', models_dir='models',
                             materialization=0):
    os.makedirs(scores_dir, exist_ok=True)
    scores_file = uid + "_scores"
    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")

    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    step_full_name = hs_previous
    for step_name, step_obj in pipeline.steps:

        step_start_time = time.time()
        step_full_name = step_full_name + str(step_obj) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        if hasattr(step_obj, 'fit_transform'):
            X_temp = step_obj.fit_transform(X_temp, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            mem_usage = memory_usage(lambda: step_obj.fit_transform(X_temp, y_train))
        elif hasattr(step_obj, 'fit'):
            mem_usage = memory_usage(lambda: step_obj.fit(X_temp, y_train))
            X_temp = step_obj.fit(X_temp, y_train)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

        if hasattr(step_obj, 'predict'):
            # step_obj.predict(X_test)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

        if random.randint(1, 100) < materialization:
            if hasattr(step_obj, 'predict'):
                artifacts.append(hs_current)
                with open(models_path, 'wb') as f:
                    pickle.dump(X_temp, f)
            else:
                artifacts.append(hs_current)
                with open(artifact_path, 'wb') as f:
                    pickle.dump(X_temp, f)

        if hs_previous == "":
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge("source", hs_current, weight=step_time, execution_time=step_time,
                                    memory_usage=max(mem_usage))
            hs_previous = hs_current
        else:
            artifact_graph.add_node(hs_current)
            artifact_graph.add_edge(hs_previous, hs_current, weight=step_time, execution_time=step_time,
                                    memory_usage=max(mem_usage))
            hs_previous = hs_current

    end_time = time.time()
    step_start_time = time.time()

    # Check if the pipeline has a classifier
    has_classifier = any(step_name == 'classifier' for step_name, _ in pipeline.steps)

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


def update_graph(artifact_graph, mem_usage, step_time, param, hs_previous, hs_current, platforms):
    artifact_graph.add_edge(hs_previous, hs_current + "_" + param, type=param, weight=step_time,
                            execution_time=step_time, memory_usage=max(mem_usage), platform=platforms)
    return hs_current + "_" + param


def extract_platform(operator):
    split_strings = operator.split('__')
    if (len(split_strings) < 2):
        return "SK"
    else:
        return split_strings[0]


def text_inside_parentheses(s):
    # Find all substrings within parentheses
    matches = re.findall(r'\((.*?)\)', s)
    # Concatenate all matches into a single string, separated by a space (or any other separator you prefer)
    return ' '.join(matches)


def extract_first_two_chars(s, selected_models=[]):
    unified_string = ''.join(selected_models)
    sig = create_4_digit_signature(text_inside_parentheses(s) + unified_string)
    split_strings = s.split('__')
    result = ''.join([substring[:2] for substring in split_strings])
    return result + sig


def create_4_digit_signature(input_string):
    # Create a hash of the input string
    hash_object = hashlib.sha256(input_string.encode())
    hex_dig = hash_object.hexdigest()

    # Convert the hexadecimal hash to an integer
    numeric_hash = int(hex_dig, 16)

    # Reduce the hash to 4 digits. We use modulo 10000 to ensure the result is at most 4 digits
    short_hash = numeric_hash % 10000

    return f"{short_hash:04}"  # Return the number as a zero-padded string


def compute_pipeline_metrics_training(artifact_graph, pipeline, uid, X_train, y_train, artifacts, mode, cc,
                                      scores_dir='metrics', artifact_dir='artifacts', models_dir='models',
                                      materialization=0):
    os.makedirs(scores_dir, exist_ok=True)
    from joblib import dump

    folder_name = "taxi_models_2"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    scores_file = uid + "_scores"
    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")

    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            fitted_operator = step_obj.fit(X_temp, y_temp)
            step_end_time = time.time()
            import copy
            fitted_operator_copy = copy.deepcopy(fitted_operator)
            if (
                    "Ra" in name or "La" in name or "KN" in name or "LG" in name or "Gr" in name or "Ri" in name or "Li" in name):
                file_path = os.path.join(folder_name, hs_current)
                with open(file_path, 'wb') as f:
                    pickle.dump(fitted_operator_copy, f)

            step_time = step_end_time - step_start_time
            cc = cc + step_time
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
            artifact_graph.add_node(hs_current + "_fit", type="fitted_operator", size=asizeof.asizeof(fitted_operator),
                                    cc=cc, frequency=1)
            fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous, hs_current,
                                                platforms)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            cc = cc + step_time
            # tmp = X_temp.__sizeof__()
            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1)
            artifact_graph.add_node(hs_current + "_ftranform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftranform",
                                       fitted_operator_name + "_super", hs_current, platforms)

    return artifact_graph, artifacts, pipeline


def compute_pipeline_metrics_evaluation(artifact_graph, pipeline, uid, X_test, y_test, artifacts):
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = "X_test__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        fitted_operator_name = hs_current + "_" + "fit"
        # print(fitted_operator_name)
        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_tetranform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetranform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time, frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                    step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ"):
                step_start_time = time.time()
                y_temp = y_temp[:len(predictions)]
                fitted_operator = step_obj.fit(y_temp)

                X_temp = fitted_operator.score(predictions)
                print(X_temp)
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms)

    return artifact_graph, artifacts, hs_previous


def compute_pipeline_metrics_training_ad(artifact_graph, pipeline, uid, X_train, y_train, artifacts, mode, cc,
                                         scores_dir='metrics', artifact_dir='artifacts', models_dir='models',
                                         materialization=0):
    os.makedirs(scores_dir, exist_ok=True)
    from joblib import dump
    folder_name = "taxi_models"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    scores_file = uid + "_scores"
    scores_path = os.path.join(scores_dir, f"{scores_file}.txt")

    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            fitted_operator = step_obj.fit(X_temp, y_temp)
            step_end_time = time.time()
            import copy
            fitted_operator_copy = copy.deepcopy(fitted_operator)
            # if ("Ra" in name or "De" in name or "KN" in name or "LG" in name or "Gr" in name or "Ri" in name or "Li" in name):
            #    file_path = os.path.join(folder_name, hs_current)
            #    with open(file_path, 'wb') as f:
            #        pickle.dump(fitted_operator_copy, f)

            step_time = step_end_time - step_start_time
            cc = cc + step_time
            if hasattr(step_obj, 'get_selected_models'):
                selected_models = step_obj.get_selected_models()
                print(selected_models)
                hs_current = extract_first_two_chars(step_full_name, selected_models)
                mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
                artifact_graph.add_node(hs_current + "_fit", type="fitted_operator",
                                        size=asizeof.asizeof(fitted_operator), cc=cc, frequency=1)
                artifact_graph.add_node(hs_current + "_Fsuper", type="super", size=0, cc=0, frequency=1)
                artifact_graph.add_edge(hs_current + "_Fsuper", hs_current + "_fit", type="super", weight=step_time,
                                        execution_time=step_time, memory_usage=max(mem_usage), platform=platforms)

                artifact_graph.add_edge(hs_previous, hs_current + "_Fsuper", type="super",
                                        weight=0,
                                        execution_time=0, memory_usage=0, platform=platforms)
                for model in selected_models:
                    artifact_graph.add_node(model + "_fit", type="fitted_operator",
                                            size=asizeof.asizeof(fitted_operator), cc=cc, frequency=1)

                    artifact_graph.add_edge(model + "_fit", hs_current + "_Fsuper", type="super",
                                            weight=0,
                                            execution_time=0, memory_usage=0, platform=platforms)
            else:
                mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
                artifact_graph.add_node(hs_current + "_fit", type="fitted_operator",
                                        size=asizeof.asizeof(fitted_operator), cc=cc, frequency=1)
                fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous,
                                                    hs_current, platforms)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            cc = cc + step_time
            # tmp = X_temp.__sizeof__()
            #### ADDING SUPER EDGE FOR TRANFORM
            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1)
            artifact_graph.add_node(hs_current + "_ftranform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftranform",
                                       fitted_operator_name + "_super", hs_current, platforms)

    return artifact_graph, artifacts, pipeline, selected_models


def compute_pipeline_metrics_evaluation_ad(artifact_graph, pipeline, uid, X_test, y_test, artifacts):
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = "X_test__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"

        hs_current = extract_first_two_chars(step_full_name)
        if hasattr(step_obj, 'get_selected_models'):
            selected_models = step_obj.get_selected_models()
            hs_current = extract_first_two_chars(step_full_name, selected_models)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        fitted_operator_name = hs_current + "_" + "fit"
        # print(fitted_operator_name)
        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_tetranform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetranform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time, frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                    step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ"):
                step_start_time = time.time()
                fitted_operator = step_obj.fit(y_temp)
                X_temp = fitted_operator.score(predictions)
                print(X_temp)
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms)

    return artifact_graph, artifacts, hs_previous


def compute_pipeline_metrics_training_helix(artifact_graph, pipeline, uid, X_train, y_train, artifacts, mode, cc,
                                            budget,
                                            scores_dir='metrics'):
    loading_speed = 566255240
    materialized_artifacts = []
    if mode == "sampling":
        hs_previous = "2sample_X_train__"
    else:
        hs_previous = "X_train__"
    X_temp = X_train.copy()
    y_temp = y_train.copy()
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:

        platforms = []
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)

        platforms.append(extract_platform(name))
        step_full_name = step_full_name + name + "__"
        hs_current = extract_first_two_chars(step_full_name)
        if hasattr(step_obj, 'fit'):
            if str(step_obj).startswith("F1ScoreCalculator"):
                continue
            if str(step_obj).startswith("AccuracyCalculator"):
                continue
            if str(step_obj).startswith("ComputeAUC"):
                continue
            if str(step_obj).startswith("KS"):
                continue
            if str(step_obj).startswith("MSECalculator"):
                continue
            if str(step_obj).startswith("MAECalculator"):
                continue
            if str(step_obj).startswith("MPECalculator"):
                continue
            step_start_time = time.time()
            y_temp = y_temp[:len(X_temp)]

            fitted_operator = step_obj.fit(X_temp, y_temp)
            step_end_time = time.time()

            step_time = step_end_time - step_start_time
            cc = cc + step_time
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.fit(X_temp, y_train))
            a_size = asizeof.asizeof(fitted_operator)
            loading_time = a_size / loading_speed
            if cc > loading_time * 2 and a_size < budget:
                budget = budget - a_size
                materialized_artifacts.append(hs_current + "_fit")

            artifact_graph.add_node(hs_current + "_fit", type="fitted_operator", size=a_size, cc=cc, frequency=1)
            fitted_operator_name = update_graph(artifact_graph, mem_usage, step_time, "fit", hs_previous, hs_current,
                                                platforms)

        if hasattr(step_obj, 'transform'):
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            cc = cc + step_time
            # tmp = X_temp.__sizeof__()
            loading_time = (X_temp.size * X_temp.itemsize) / loading_speed
            if cc > loading_time * 2 and (X_temp.size * X_temp.itemsize) < budget:
                budget = budget - asizeof.asizeof(X_temp.size * X_temp.itemsize)
                materialized_artifacts.append(hs_current + "_ftranform")

            artifact_graph.add_node(fitted_operator_name + "_super", type="super", size=0, cc=0, frequency=1)
            artifact_graph.add_node(hs_current + "_ftranform", type="train", size=X_temp.size * X_temp.itemsize, cc=cc,
                                    frequency=1)
            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_super", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "ftranform",
                                       fitted_operator_name + "_super", hs_current, platforms)

    return artifact_graph, artifacts, pipeline, materialized_artifacts, budget

def compute_pipeline_metrics_evaluation_helix(artifact_graph, pipeline, uid, X_test, y_test, artifacts,
                                              materialized_artifacts, budget):
    loading_speed = 566255240
    X_temp = X_test.copy()
    y_temp = y_test.copy()
    hs_previous = "X_test__"
    step_full_name = hs_previous
    fitted_operator_name = ""
    for step_name, step_obj in pipeline.steps:
        platforms = []
        platforms.append(extract_platform(str(step_obj)))
        if "GP" in str(step_obj) or "TF" in str(step_obj) or "TR" in str(step_obj) or "GL" in str(step_obj) in str(
                step_obj):
            name = str(step_obj)
        else:
            name = "SK__" + str(step_obj)
        step_full_name = step_full_name + str(name) + "__"
        hs_current = extract_first_two_chars(step_full_name)
        # artifact_path = os.path.join(artifact_dir, f"{hs_current}.pkl")
        # models_path = os.path.join(models_dir, f"{hs_current}.pkl")
        fitted_operator_name = hs_current + "_" + "fit"
        # print(fitted_operator_name)
        if hasattr(step_obj, 'transform'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.transform(X_temp))
            step_start_time = time.time()
            X_temp = step_obj.transform(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            loading_time = (X_temp.size * X_temp.itemsize) / loading_speed
            if cc > loading_time * 2 and (X_temp.size * X_temp.itemsize) < budget:
                budget = budget - asizeof.asizeof(X_temp.size * X_temp.itemsize)
                materialized_artifacts.append(hs_current + "_tetranform")

            artifact_graph.add_node(hs_current + "_tetranform", type="test", size=X_temp.size * X_temp.itemsize,
                                    cc=cc + step_time, frequency=1)
            artifact_graph.add_node(fitted_operator_name + "_Tsuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Tsuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "tetranform",
                                       fitted_operator_name + "_Tsuper", hs_current, platforms)

        if hasattr(step_obj, 'predict'):
            cc = artifact_graph.nodes[fitted_operator_name]['cc']
            step_start_time = time.time()
            predictions = step_obj.predict(X_temp)
            step_end_time = time.time()
            step_time = step_end_time - step_start_time

            loading_time = (predictions.size * predictions.itemsize) / loading_speed
            if cc > loading_time * 2 and (predictions.size * predictions.itemsize) < budget:
                budget = budget - asizeof.asizeof(predictions.size * predictions.itemsize)
                materialized_artifacts.append(hs_current + "_predict")

            artifact_graph.add_node(hs_current + "_predict", type="test", size=predictions.size * predictions.itemsize,
                                    cc=cc + step_time, frequency=1)

            artifact_graph.add_node(fitted_operator_name + "_Psuper", type="super", size=0, cc=0, frequency=1)

            artifact_graph.add_edge(fitted_operator_name, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)
            artifact_graph.add_edge(hs_previous, fitted_operator_name + "_Psuper", type="super", weight=0,
                                    execution_time=0, memory_usage=0, platform=platforms)

            mem_usage = [0, 0]  # memory_usage(lambda: step_obj.predict(X_temp ))
            hs_previous = update_graph(artifact_graph, mem_usage, step_time, "predict",
                                       fitted_operator_name + "_Psuper", hs_current, platforms)
        if hasattr(step_obj, 'score'):
            if str(step_obj).startswith("F1ScoreCalculator") or str(step_obj).startswith("AccuracyCalculator") or str(
                    step_obj).startswith("MPECalculator") or str(step_obj).startswith("MSE") or str(
                    step_obj).startswith("MAE") or str(step_obj).startswith("KS") or str(step_obj).startswith("VIZ"):

                step_start_time = time.time()
                y_temp = y_temp[:len(predictions)]
                fitted_operator = step_obj.fit(y_temp)

                X_temp = fitted_operator.score(predictions)
                print(X_temp)
                step_end_time = time.time()
                step_time = step_end_time - step_start_time

                cc = artifact_graph.nodes[hs_previous]['cc']

                loading_time = (X_temp.size * X_temp.itemsize) / loading_speed
                if cc > loading_time * 2 and (X_temp.size * X_temp.itemsize) < budget:
                    budget = budget - asizeof.asizeof(X_temp.size * X_temp.itemsize)
                    materialized_artifacts.append(hs_current + "_score")

                artifact_graph.add_node(hs_current + "_score", type="score",
                                        size=X_temp.size * X_temp.itemsize, cc=cc, frequency=1)

                hs_previous = update_graph(artifact_graph, mem_usage, step_time, "score", hs_previous, hs_current,
                                           platforms)

    return artifact_graph, artifacts, hs_previous, materialized_artifacts
