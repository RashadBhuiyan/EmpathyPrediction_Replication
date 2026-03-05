# EmpathyPrediction_Replication

Disclaimer: This README was created with the assistance of an AI tool, GPT-5 mini through VS Code. Estimated carbon cost for the two queries that produced this content is approximately 0.8–8 g CO2e (assumes 0.001–0.01 kWh per query and a grid carbon intensity of 0.4 kg CO2e/kWh).

**Project**

This repository replicates experiments from the paper "Empathy Prediction from Diverse Perspectives" (<https://aclanthology.org/2025.acl-long.439/>) (Chen et al., ACL 2025). The project implements the data processing, model variants, and evaluation setup described in that work to reproduce and analyze empathy prediction results across diverse perspectives.

**Dataset**

The `datasets/` folder contains the data used for replication. Files included in this repository are:

- [datasets/anon_efp_train.csv](datasets/anon_efp_train.csv) — anonymized training split from the original paper
- [datasets/anon_efp_val.csv](datasets/anon_efp_val.csv) — anonymized validation split from the original paper
- [datasets/anon_efp_test.csv](datasets/anon_efp_test.csv) — anonymized test split from the original paper
- [datasets/PAIRS_train.csv](datasets/PAIRS_train.csv) — PAIRS training data (sourced from the empathic-stories collection)
- [datasets/PAIRS_dev.csv](datasets/PAIRS_dev.csv) — PAIRS development data
- [datasets/PAIRS_test.csv](datasets/PAIRS_test.csv) — PAIRS test data

Notes:
- The `anon_` files are taken from the original paper's released data.
- The `PAIRS_*` files are drawn from the MIT Media Lab `empathic-stories` dataset: https://github.com/mitmedialab/empathic-stories

**Model breakdown / approach**

The replication follows the modeling approach described in Section 4 of the paper; please consult Section 4 for the full method and architecture details. Appendix D provides the exact LLM prompts used in experiments, and Appendix E contains additional information about model setup and training hyperparameters.

**Evaluation metrics and setup**

Evaluation follows the protocol in Section 5 of the paper. The primary metrics reported (Section 5.1) include:

- Spearman correlation
- Pearson correlation
- Accuracy
- F1 score
- Precision
- Recall
- Mean-squared error (MSE)

Refer to Section 5 of the paper for exact evaluation splits, aggregation procedures, and any task-specific thresholding or binarization applied before metric calculation.

