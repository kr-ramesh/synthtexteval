### Privacy Metrics 

- **Entity Leakage**: Measures the unintended inclusion of private or sensitive entities from the training data in generated synthetic text, posing privacy risks.

- **Entity-based Span Memorization**: Identifies when spans of text that include sensitive entities, are directly regurgitated from the training data, indicating potential memorization by the model.

- **Canary Evaluations**: A method for detecting memorization in language models by inserting specific "canary" tokens or phrases into the training data and measuring the exposure of these canaries.


### Related Material and Citations

```
@inproceedings {236216,
author = {Nicholas Carlini and Chang Liu and {\'U}lfar Erlingsson and Jernej Kos and Dawn Song},
title = {The Secret Sharer: Evaluating and Testing Unintended Memorization in Neural Networks},
booktitle = {28th USENIX Security Symposium (USENIX Security 19)},
year = {2019},
isbn = {978-1-939133-06-9},
address = {Santa Clara, CA},
pages = {267--284},
url = {https://www.usenix.org/conference/usenixsecurity19/presentation/carlini},
publisher = {USENIX Association},
month = aug
}
```