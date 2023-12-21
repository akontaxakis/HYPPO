 HYPPO: Using Equivalences to Optimize Pipelines in Exploratory Machine Learning
---
This public repository contains the main functionality of the Hyppo System, in addition it also contrains a pipeline generator. 

![HYPPO’s system architecture](https://onedrive.live.com/embed?resid=883817D68644DE5%2125087&authkey=%21AORdRWjhTobPdrQ&width=1586&height=859)

---
## Hyppo Components
We implemented HYPPO on top of the sklearn.pipelines API and networkX
 ### Dictionary 
he HYPPO's dictionary involved all the physical operators that we have implemented. For extending the dictonary make sure you implement the core functions:
- fit: This function is for training or fitting your operator to the data. It's where any initial analysis or setup specific to your operator should be done.
- transform: This function is used for the transformation or processing of the data using your operator. Whatever unique operations your operator is designed to perform will be coded here.
- score: If your operator involves any form of evaluation or scoring, this function should be implemented. It's used to assess the performance or output of your operator.
Ensure Full HYPPO Compatibility: By implementing these functions, your custom operators will be compatible with HYPPO's features and functionalities, allowing them to seamlessly integrate with the rest of the system.
<!---
 ### Parser:
 ### Augmenter:
 ### History Manager:
-->

## Pipeline generator

The pipeline generator is not part of the HYPPO system, it was created to for testing and evaluating purposes. The pipeline generator gets as input a dataset and a pool of operators and 
generates a sequence
of pipelines for execution. A possible pipeline is annotated with an operator id and the operators are seperated by the pipe: |

```
"SI|SS|SVM()|F1" -> Example of a pipeline 
```

The example above is referring to a pipeline with the following steps Imputation->Scaler->SVM-> F1 Score. After a pipeline is expressed we 
let our generator randomly select a physical implementation for each operator. 

 ### History
The output of our pipeline generator is a history. After the execution of each pipeline the history graph is dropped into a file in the directory ```/graphs/iterative```.
By executing _N_ pipelines _N_ graphs will be dropped. 

##  Plan generator
the Plan generator was created as a separate project at https://github.com/akontaxakis/Plan-Generator

## Getting Started

- git clone https://github.com/akontaxakis/HYPPO.git

### Contact

For any questions don't hesitate to ask:

Antonios Kontaxakis, antonios.kontaxakis-ATNOSPAM-ulb.be
