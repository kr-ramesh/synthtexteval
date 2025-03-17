from syntheval.descriptive.compare import basic_comparison_metrics, compare_distributions
t1 = [
    "This is a short sentence.",
    "Here is another one.",
    "This one is a bit longer than the others."
]

t2 = [
    "This is the first text.",
    "Here comes the second one.",
    "The third text is somewhat longer compared to the first two."
]

# Compare basic metrics
basic_comparison_metrics(t1, t2)
# Compare distributions using various metrics
compare_distributions(t1, t2, ['kl_divergence', 'jaccard', 'cosine'])

