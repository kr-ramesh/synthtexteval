### SynthEval: A Toolkit for Generating and Evaluating Synthetic Data Across Domains

<a href="https://colab.research.google.com/drive/1pM3Y0DGYAY2ocSismwOmshIJlWwC5WQW?usp=sharing" alt="Colab">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" /></a>

### Contents
- [SynthEval: A Toolkit for Generating and Evaluating Synthetic Data Across Domains](#syntheval-a-toolkit-for-generating-and-evaluating-synthetic-data-across-domains)
    - [Contents](#contents)
    - [Introduction to SynthEval](#introduction)
      - [Overview](#overview)
      - [Key Features of SynthEval](#key-features-of-syntheval)
      - [Jupyter Notebook](#jupyter-notebooks)
    - [Repository Structure](#repository-structure)
    - [Installation Instructions](#installation-instructions)
      - [Setting up the environment](#setting-up-the-environment)
    - [Evaluation Pipeline](#evaluation-pipeline)
        - [Training the Model to Generate Synthetic Data](#training-the-model-to-generate-synthetic-data)
            - [Using Differential Privacy to Generate Synthetic Data](#using-differential-privacy-to-generate-synthetic-data) 
        - [Generating Synthetic Data](#generating-synthetic-data)
        - [Generating Descriptive Statistics](#generating-descriptive-statistics)
        - [Evaluating Downstream Utility](#evaluating-downstream-utility)
        - [Fairness Evaluation](#fairness-evaluation)
        - [Privacy Evaluation](#privacy-evaluation)
        - [Qualitative Evaluation](#qualitative-evaluation)
    - [Citations](#citations)

### Installation Instructions:

1. Clone the repository (the command below is for public repositories only)
```
git clone https://github.com/kr-ramesh/sdg-eval/
```

2. Execute the following commands to install the dependencies for the environment to use the package and install it locally (Note: Need to publish this package to pip):
```
pip install -r requirements.txt
pip install -e .
```

3. Downloading data to test the toolkit (PhysioNet)

### Evaluation Pipeline:

Sample usage:


```
data = {
    'texts': [
        ...
    ]
}
df = pd.DataFrame(data)
```

### Training the Model to Generate Synthetic Data

#### Using Differential Privacy to Generate Synthetic Data

### Generating Synthetic Data

### Generating Descriptive Statistics

```
from sdgeval.descriptive.descriptor import Descriptor
from sdgeval.descriptive.arguments import DescriptorArgs
desc_analyze = Descriptor(data['texts'], DescriptorArgs(produce_plot=True))
```

### Evaluating Downstream Utility

### Fairness Evaluation

```
p_df, f_df = analyze_group_fairness_performance(df, problem_type = problem_type, num_classes = n)
```

### Privacy Evaluation

### Qualitative Evaluation

```
from sdgeval.qual.arguments import MauveArgs, LMArgs, FrechetArgs
from sdgeval.qual.mauve_metric import calculate_mauve_score
from sdgeval.qual.frechet import calculate_fid_score
from sdgeval.qual.perplexity import calculate_perplexity

result = calculate_mauve_score(df, MauveArgs)

result = calculate_fid_score(df, FrechetArgs)

result = calculate_perplexity(df, LMArgs)
```

