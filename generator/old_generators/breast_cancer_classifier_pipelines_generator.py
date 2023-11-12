# This is a sample Python script.
# Press the green button in the gutter to run the script.
# Generate a random pipeline with 1 to 5 steps
import uuid

import networkx as nx
from sklearn.linear_model import Lasso, LogisticRegression

from generator.pipelines_steps.steps_examples import simple_eq_steps
from libs.Pipelines_Library import generate_pipeline, compute_pipeline_metrics, \
    fit_pipeline_with_store_or_load_artifacts
from libs.artifact_graph_lib import plot_artifact_graph, store_or_load_artifact_graph, create_equivalent_graph
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    import os
    os.chdir("C:/Users/adoko/PycharmProjects/pythonProject1")


    # Load the Breast Cancer Wisconsin dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    #('scaler', [StandardScaler(), MinMaxScaler()]),
    #('imputer', [SimpleImputer(strategy='mean'), SimpleImputer(strategy='median')]),
    # Define the steps
    steps = simple_eq_steps
    sum = 0;
    uid = str(uuid.uuid1())[:8]
    number_of_steps = len(steps);
    artifact_graph = nx.DiGraph()
    artifact_graph.add_node("source")
    # Generate and execute 100 pipelines
    for i in range(20):
        pipeline = generate_pipeline(steps, number_of_steps)
        if pipeline == None:
            print("no model")
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Check if the pipeline has a classifier,
                has_classifier = any(step_name == '3.classifier' for step_name, _ in pipeline.steps)

                if has_classifier:
                    pipeline.fit(X_train, y_train)
                    score = pipeline.score(X_test, y_test)
                    sum = sum + 1
                else:
                    score = None

                step_times, artifact_graph = compute_pipeline_metrics(artifact_graph, pipeline,uid, X_train, X_test,
                                                                      y_train, y_test)
                artifacts = fit_pipeline_with_store_or_load_artifacts(pipeline, X_train ,y_train,30)

                print(f"Pipeline {i + 1}:\n{pipeline}\nScore: {score}\n")
            except TypeError:
                print("Oops!  Wrong Type.  Try again...")
            except ValueError:
                print("Oops!  That was no valid number.  Try again...")
    print(sum)
    store_or_load_artifact_graph(artifact_graph,sum,uid,'classifier','breast_cancer')
    plot_artifact_graph(artifact_graph,uid)
    equivalent_graph = create_equivalent_graph(steps, uid, artifact_graph, metrics_dir='metrics')
    plot_artifact_graph(equivalent_graph, "eq_"+uid)
    store_or_load_artifact_graph(equivalent_graph, sum, uid, 'eq_classifier', 'breast_cancer')

