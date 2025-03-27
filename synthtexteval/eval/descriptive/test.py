from synthtexteval.eval.descriptive.compare import basic_comparison_metrics, compare_distributions
from synthtexteval.eval.descriptive.descriptor import TextDescriptor
from synthtexteval.eval.descriptive.arguments import TextDescriptorArgs
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

desc = TextDescriptor(texts = t1, args = TextDescriptorArgs(produce_plot=True), reference_texts = t2)
tm = desc._topic_modeling(num_topics=3)
print(tm)
