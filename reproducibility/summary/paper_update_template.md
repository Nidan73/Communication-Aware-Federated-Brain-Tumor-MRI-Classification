# Multi-seed Paper Update Template

Do not replace placeholders until all four new runs, prediction validation, summary generation, and paired statistical analysis have completed.

## 1. Methodology paragraph

To assess sensitivity to random initialization and stochastic optimization, we repeated the principal Non-IID FedAvg comparison with three random seeds (42, 123, and 2025) for EfficientNet-B0 and ResNet50. Each experiment used four simulated clients, 20 communication rounds, three local epochs per round, batch size 32, AdamW (learning rate 0.0003; weight decay 0.0001), and full client participation. For each seed, both backbones loaded the same saved 70/30 label-skew client assignment, while different seeds used independently generated assignments. Every model began from its architecture-specific ImageNet-pretrained weights and not from the centralized brain-MRI checkpoint. The best global round was selected using validation macro-F1, after which the fixed test set was evaluated once. Full-state model upload and download were counted for all four clients in every round. Python, NumPy, PyTorch CPU/CUDA, DataLoader, and worker seeds were controlled; however, perfect bit-for-bit reproducibility is not guaranteed across different GPU, CUDA, cuDNN, PyTorch, driver, operating-system, or hardware versions.

## 2. Results paragraph

Across seeds 42, 123, and 2025, FedAvg EfficientNet-B0 achieved a test accuracy of <EFFNET_MEAN_ACC> ± <EFFNET_SD_ACC> and a macro-F1 of <EFFNET_MEAN_F1> ± <EFFNET_SD_F1> (mean ± sample standard deviation). FedAvg ResNet50 achieved a test accuracy of <RESNET_MEAN_ACC> ± <RESNET_SD_ACC> and a macro-F1 of <RESNET_MEAN_F1> ± <RESNET_SD_F1>. EfficientNet-B0 selected a mean best round of <EFFNET_MEAN_BEST_ROUND> with range <EFFNET_BEST_ROUND_RANGE>; ResNet50 selected a mean best round of <RESNET_MEAN_BEST_ROUND> with range <RESNET_BEST_ROUND_RANGE>. These values were calculated from full-precision seed-level metrics and rounded only for presentation.

## 3. Statistical-analysis paragraph

Predictions were paired by stable sample identifier for all 1,440 fixed test images. For seed <SEED>, the exact two-sided McNemar test gave p = <MCNEMAR_P>, with <ONLY_EFFNET_CORRECT> images correct only for EfficientNet-B0 and <ONLY_RESNET_CORRECT> correct only for ResNet50; the paired difference was therefore <MCNEMAR_INTERPRETATION> on the fixed test set. A paired bootstrap with <BOOTSTRAP_RESAMPLES> image-level resamples and a fixed bootstrap seed estimated the EfficientNet-B0-minus-ResNet50 accuracy difference as <ACC_DIFFERENCE> with 95% percentile CI <BOOTSTRAP_ACC_CI>, and the macro-F1 difference as <F1_DIFFERENCE> with 95% percentile CI <BOOTSTRAP_F1_CI>. The <METRIC> confidence interval <INCLUDES_OR_EXCLUDES_ZERO>. Across the three random seeds, the mean paired accuracy difference was <MEAN_SEED_ACC_DIFFERENCE> (sample SD <SD_SEED_ACC_DIFFERENCE>) and the mean paired macro-F1 difference was <MEAN_SEED_F1_DIFFERENCE> (sample SD <SD_SEED_F1_DIFFERENCE>). Because only three seed-level pairs were available, no strong significance or superiority claim was based on a seed-level hypothesis test.

## 4. Updated limitations paragraph

The multi-seed analysis reduces uncertainty arising from a single stochastic run but remains limited to three seeds, one fixed leakage-controlled split, four simulated clients, and one 70/30 label-skew construction. Image-level McNemar and bootstrap results characterize paired behavior on the fixed 1,440-image test set and do not by themselves establish general superiority across institutions, acquisition protocols, populations, or alternative federation patterns. Hardware and software differences can also prevent bit-for-bit replication despite controlled random-number generators. Future work should include more seed-level repetitions, external multi-institutional validation, additional heterogeneity regimes, and privacy- or systems-realistic federated deployments.

## 5. Compact LaTeX table template

```latex
\begin{table}[t]
\centering
\caption{Three-seed Non-IID FedAvg results (mean $\pm$ sample SD; seeds 42, 123, and 2025).}
\label{tab:multiseed_noniid}
\begin{tabular}{lcccc}
\toprule
Model & Accuracy (\%) & Macro-F1 (\%) & State (MiB) & Communication (MiB) \\
\midrule
EfficientNet-B0 & <EFFNET_MEAN_ACC> $\pm$ <EFFNET_SD_ACC> & <EFFNET_MEAN_F1> $\pm$ <EFFNET_SD_F1> & <EFFNET_STATE_MIB> & <EFFNET_COMM_MIB> \\
ResNet50 & <RESNET_MEAN_ACC> $\pm$ <RESNET_SD_ACC> & <RESNET_MEAN_F1> $\pm$ <RESNET_SD_F1> & <RESNET_STATE_MIB> & <RESNET_COMM_MIB> \\
\bottomrule
\end{tabular}
\end{table}
```

