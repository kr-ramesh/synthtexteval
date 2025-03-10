### Fairness Metrics

We used the notions of Equalized Odds and Equality Difference measures to assess fairness. 


- **Equalized Odds**: Equalized Odds (EO) is a fairness criterion that informs us whether predictive model's errors are equally distributed across different demographic groups. A model satisfies EO if the true positive rate (TPR) and false positive rate (FPR) are the same across all groups. In other words, the model should not be biased toward or against any particular group in terms of the likelihood of correct or incorrect predictions.

- **Equality Difference**: Equality Difference (ED) metrics measure the disparity in performance rates between different subgroups. For example, the False Positive Equality Difference (FPED) measures the disparity in the false positive rate (FPR) between different groups. A lower FPED means that the model's false positives are more balanced across groups, ensuring that no group is disproportionately affected by false positive errors. Similarly, we define the ED metrics for true positives, true negative and false negatives.

- **F1 Difference**: F1 Difference is a fairness metric that evaluates the disparity in F1 scores across different demographic or protected groups. The F1 score is the harmonic mean of precision and recall, and the macro average computes the F1 score for each class individually and then averages them, while the micro average computes the overall metrics by aggregating the contributions of all classes, treating each individual prediction equally regardless of the class. The F1 difference measures the difference in the max and the minimum F1 scores between different subgroups. A lower F1 difference indicates that the model's performance is balanced, and a large fifference suggests that the model performs better on some groups while underperforming on others, indicating potential unfairness in the system's predictions.






