Installation instructions:

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


Sample usage:


```
data = {
    'texts': [
        ...
    ]
}
df = pd.DataFrame(data)
```

- Descriptive evaluation

```
from sdgeval.descriptive.descriptor import Descriptor
from sdgeval.descriptive.arguments import DescriptorArgs
desc_analyze = Descriptor(data['texts'], DescriptorArgs(produce_plot=True))
```

- Downstream evaluation

- Fairness evaluation

```
p_df, f_df = analyze_group_fairness_performance(df, problem_type = problem_type, num_classes = n)
```

- Privacy evaluation

- Qualitative evaluation (requires an update as I've made modifications, but this is still functional)

```
from sdgeval.qual.arguments import MauveArgs, LMArgs, FrechetArgs
from sdgeval.qual.mauve_metric import calculate_mauve_score
from sdgeval.qual.frechet import calculate_fid_score
from sdgeval.qual.perplexity import calculate_perplexity

result = calculate_mauve_score(df, MauveArgs)

result = calculate_fid_score(df, FrechetArgs)

result = calculate_perplexity(df, LMArgs)
```

