# Initialize the elements
import itertools
import random
from sklearn.pipeline import Pipeline

from Example.user_iterations import collab_HIGGS_all_operators


def create_pipelines_for_combinations(shuffled_combinations, operators_dict):
    all_pipelines = []
    for combination_string in shuffled_combinations:
        # Split the combination string by commas to get individual step names
        combination = combination_string.split(', ')
        steps = []
        for step_name in combination:
            operators = operators_dict.get(step_name)

            # If there are multiple operators, choose one. This can be random or based on some selection logic.
            if operators and len(operators) > 1:
                operator = random.choice(operators)  # Random selection
            elif operators:
                operator = operators[0]  # If there's only one, choose it.
            else:
                continue  # If no operator matches, skip adding this step.

            steps.append((step_name, operator))

        # Create a pipeline for this combination
        pipeline = Pipeline(steps=steps)
        all_pipelines.append(pipeline)
    return all_pipelines

def get_pipeline_signature(pipeline):
    # Create a simplified representation of a pipeline based on the class names of its steps and main parameters
    signature = []
    for step_name, step_process in pipeline.steps:
        step_signature = step_name + "_" + step_process.__class__.__name__

        # Add main parameters to the signature (modify this based on what you consider 'main')
        if hasattr(step_process, 'C'):  # for SVM, for instance
            step_signature += f"_C={step_process.C}"
        if hasattr(step_process, 'degree'):  # for PolynomialFeatures, for instance
            step_signature += f"_degree={step_process.degree}"
        # Extend with other parameters as needed

        signature.append(step_signature)

    return '|'.join(signature)

def create_pipelines_for_combinations(shuffled_combinations, operators_dict, mode='single'):
    all_pipelines = []
    seen_signatures = set()  # Store signatures of pipelines we've already created

    for combination_string in shuffled_combinations:
        combination = combination_string.split(', ')

        if mode == 'all_physical_pipelines':
            all_combination_operators = []

            for step_name in combination:
                operators = operators_dict.get(step_name, [])
                all_combination_operators.append(operators)

            for operators_combination in itertools.product(*all_combination_operators):
                steps = [(step_name, operator) for step_name, operator in zip(combination, operators_combination)]
                pipeline = Pipeline(steps=steps)

                signature = get_pipeline_signature(pipeline)
                if signature not in seen_signatures:  # Only add this pipeline if we haven't seen this signature before
                    seen_signatures.add(signature)
                    all_pipelines.append(pipeline)

        else:  # Default mode, 'single'
            steps = []
            for step_name in combination:
                operators = operators_dict.get(step_name, [])

                if operators and len(operators) > 1:
                    operator = random.choice(operators)  # Random selection
                elif operators:
                    operator = operators[0]  # If there's only one, choose it.
                else:
                    continue  # If no operator matches, skip adding this step.
                steps.append((step_name, operator))
            # Create a pipeline for this combination if there are any steps
            if steps:
                pipeline = Pipeline(steps=steps)
                all_pipelines.append(pipeline)

    return all_pipelines
def print_pipelines(pipelines):
        for i, pipeline in enumerate(pipelines):
            print(f"Pipeline {i + 1}:")
            for step_name, step_process in pipeline.steps:
                # Here, we're only printing the step name and the class name of the operator for brevity.
                print(f"- {step_name}: {step_process.__class__.__name__}")
            print("\n")  # Add a newline for better readability
def shuffle_combinations(combinations):
    random.shuffle(combinations)
    return combinations

def string_to_elements(param_string):
    list_strings = param_string.split('|')
    return [element.split(';') for element in list_strings]

def generate_combinations(param_string):
        # Parse the parameter string back to lists
        elements = string_to_elements(param_string)

        # Generate combinations
        combinations = []

        # Use itertools.product, it calculates the cartesian product of input iterables,
        # which is equivalent to nested for-loops in a generator expression.
        for combination in itertools.product(*elements):
            combinations.append(", ".join(filter(None, combination)))  # filter is used to exclude empty strings

        return combinations


def logical_to_physical_random(param_string,operators_dict,n, mode = 'single' ):
    # Generate and print all the combinations
    all_combinations = generate_combinations(param_string)
    shuffled_combinations = shuffle_combinations(all_combinations)
    selected_logical_pipelines = [random.choice(shuffled_combinations) for _ in range(n)]
    # Create pipelines for each combination
    all_pipelines = create_pipelines_for_combinations(selected_logical_pipelines, operators_dict)
    # Print out summaries of all pipelines
    #print_pipelines(all_pipelines)
    return all_pipelines

def logical_to_physical(param_string,operators_dict, mode = 'single'):
    # Generate and print all the combinations
    all_combinations = generate_combinations(param_string)
    shuffled_combinations = shuffle_combinations(all_combinations)
    # Create pipelines for each combination
    all_pipelines = create_pipelines_for_combinations(shuffled_combinations, operators_dict,mode)
    # Print out summaries of all pipelines
    #print_pipelines(all_pipelines)
    return all_pipelines

if __name__ == '__main__':
    operators_dict = {key: value for key, value in collab_HIGGS_all_operators}
    param_string = "SI|;SS|;PF|SVM(0.5);SVM(0.05);SVM(0.005)|AC;F1"
    pipelines = logical_to_physical(param_string,operators_dict, 'all_physical_pipelines')
    print_pipelines(pipelines)