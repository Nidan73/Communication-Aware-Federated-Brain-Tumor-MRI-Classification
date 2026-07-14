# Asset Index

These assets are intended for a compact 4-6 page conference paper. All tables and figures were regenerated from completed 20-round result files and saved outputs only. No model training or federated learning rerun was performed.

## Tables

- `table_centralized_screening.md` / `table_centralized_screening.csv`  
  Use in the centralized screening subsection. Values come from the saved centralized validation benchmark; each row is the best predefined configuration for that backbone.

- `table_final_federated_results.md` / `table_final_federated_results.csv`  
  Use as the main federated results table. The primary Non-IID FedAvg rows report three-seed mean $\pm$ sample SD; IID and FedProx rows are labeled single-seed support.

- `table_main_communication_comparison.md` / `table_main_communication_comparison.csv`  
  Use near the communication discussion. This compact table compares only FedAvg EfficientNet-B0 Non-IID against FedAvg ResNet50 Non-IID under the same 4-client, 20-round setting and reports full-state payload per run.

## Figures

- `fig_main_communication_comparison.png` / `fig_main_communication_comparison.pdf`  
  Use as the main full-state-payload figure. It focuses on the paper's primary model-size and communication comparison.

- `fig_multiseed_convergence.png` / `fig_multiseed_convergence.pdf`  
  Use to show mean validation macro-F1 across seeds 42, 123, and 2025; shading denotes $\pm1$ sample SD.

- `fig_convergence_val_macro_f1.png` / `fig_convergence_val_macro_f1.pdf`  
  Legacy single-seed convergence asset retained for provenance; do not use as the primary paper figure.

- `fig_confusion_matrix_efficientnet_b0_noniid.png` / `fig_confusion_matrix_efficientnet_b0_noniid.pdf`  
  Use as the main confusion matrix for the selected compact Non-IID federated model.

## Text

- `results_paragraphs.md`  
  Contains cautious, paper-ready results wording suitable for the Results and Discussion sections. It avoids state-of-the-art claims and includes the required EfficientNet-B0 vs ResNet50 communication comparison.
