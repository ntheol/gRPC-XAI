# Explainability Module
The Explainability Module component is used to enhance the trustworthiness of the model. <be>
It extends the original explainability algorithms to provide explanations for the experimental process and the hyperparameters of the trained ML model. 

UNSW_NB15.ipynb notebook contains an example of using the Explainability Module via gRPC framework for the transfer of data between a client and a server.
The notebook contains the whole process of dataset preprocessing and model training and then the use of the different Explanation Functions.

Right now we have implemented on our module the below Explainability Algorithms:

For Pipeline Explanations:
- PDPlots
- ALEPlots
- InfluenceFunctions
- CounterfactualExplanations
  
For Model Explanations:
- PDPlots
- ALEPlots(in progress)
# Setup

## Download Source Code

```shell
git clone https://github.com/ntheol/gRPC-XAI.git
```
### Create virtual enviroment
```shell
# using pip
pip install -r requirements.txt

# using Conda
conda create --name <env_name> --file requirements.txt
```
# Explainability Module Workflow

## 1. Train an ML Model
Given a labeled dataset, an ML pipeline is trained using standard methodologies.

## 2. Choose Explainability Type
Users select the type of explainability they desire:
- **Pipeline**: Provides insights into the experimental process and the hyperparameters of the model.
- **Model**: Offers explanations about the behavior of the ML model and how dataset features influence model predictions.

## 3. Choose Explainability Method
Users then select the specific method for obtaining explanations:
- **Global Methods**:
  - **Partial Dependence (PD) Plots**: Illustrate the marginal effect of one or two features on the predicted outcome.
  - **Accumulated Local Effects (ALE) Plots**: Reveal the average effect of one or two features on the predicted outcome, incorporating interactions with other features.
  
  *For Pipeline Explainability Type*: A surrogate model is trained with different hyperparameter configurations and their respective accuracy.

- **Local Methods**:
  - **Counterfactuals**: For a misclassified test instance, generates alternative scenarios where changes in hyperparameters would lead to correct predictions.
  - **Influence Functions**: Determines how much each training sample influences errors on selected test instances, particularly useful for identifying influential data points.
  
  *For Pipeline Explainability Type*: A surrogate model is trained with different hyperparameter configurations and their respective predictions on the misclassified test instance.

