### Descriptive Functions
- _get_least_frequent_entities(), _get_entity_count(), _get_top_n_entities(), _plot_entity_frequency(), _ngram_frequency()
- _compute_tfidf()
- _topic_modeling()

### Arguments
We provide the following arguments for our TextDescriptor class:

| Argument | Argument Type   | Description   |
| :---:   | :---: | :--- |
| ```pkl_file_path``` | str  | Specifies the file path where the analysis of the entity results will be saved in a pickle format.   |
| ```plt_file_path``` | str  | Specifies the file path where the plot of the entities sorted by frequency will be saved in PNG format. |
| ```min_threshold``` | int  | Defines the lower bound threshold for returning the least frequent entities. Entities with a frequency below this value will be included in the results. |
| ```max_threshold``` | int  | Description: Defines the upper bound threshold for returning the most frequent entities. Entities with a frequency above this value will be included in the results. |
| ```produce_plot``` | bool  | If set to True, a plot of the most frequent entities will be generated and saved. If set to False, no plot will be produced. |