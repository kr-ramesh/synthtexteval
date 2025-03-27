from synthtexteval.eval.downstream.classify.generate_silver_annotations import generate_silver_annotations

generate_silver_annotations(
    model_name = "bert-base-uncased",
    path_to_model = "/export/fs06/kramesh3/psd/multilabel-models/princeton_mimic_10ICD_DP_inf/princeton_mimic_10ICD_DP_inf_v0",
    n_labels = 10,
    problem_type = 'multi_label_classification',
    data_path = "/export/fs06/kramesh3/old-files/misc-files/data/princeton_mimic_10ICD_DP_inf/eval.csv",
    text_column = "TEXT",
    label_column = "Label",
    output_path = "test_annotated.csv",
    ckpt_exists = True
)