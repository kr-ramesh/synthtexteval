### Arguments

We provide the following arguments for training and generating synthetic data with the controllable generation approach:

| Argument | Argument Type   | Description   |
| :---:   | :---: | :--- |
| ```disable_dp``` | bool | Indicates whether differential privacy (DP) should be disabled. If set, DP is disabled, otherwise it is enabled. |
| ```target_epsilon``` | float | Specifies the target epsilon value for differential privacy, controlling the privacy budget for the model's training process. |
| ```target_delta``` | float | Defines the target delta value for differential privacy, controlling the probability of a privacy breach occurring. |
| ```max_grad_norm``` |  float | Specifies the maximum gradient norm for gradient clipping, helping to prevent excessively large updates during training. | 
| ```enable_lora``` | bool | Determines whether LoRA (Low-Rank Adaptation) should be enabled for the model. If True, LoRA is applied, otherwise it is disabled. |
| ```lora_dim``` |  int | Sets the dimensionality of the low-rank adaptation matrix for LoRA, controlling the size of the parameters being adapted. |
| ```lora_alpha``` |  float | Controls the scaling factor for LoRA, adjusting how much influence the low-rank adaptation has on the model's weights. |
| ```lora_dropout``` |  float | Specifies the dropout rate for LoRA layers, used to prevent overfitting. |
