# Final Anonymous COMPAS Submission Report

1. **Final paper title:** Communication-Aware Federated Brain Tumor MRI Classification.

2. **Final PDF path:** `paper_anonymous_submission/main.pdf`.

3. **Final page count:** 6 pages, US Letter, IEEE conference layout.

4. **Pipeline source:** author-supplied vector PDF `figures/federated_mri_pipeline.pdf`. The visible text, values, shapes, and layout were preserved.

5. **Pipeline script status:** `scripts/generate_final_pipeline.py` remains as a historical generator but was not used for the active figure.

6. **Pipeline PDF dimensions:** 1127.04 × 379.92 points, one vector page.

7. **Pipeline SVG status:** the older SVG is retained as an inactive historical asset; the manuscript includes the supplied PDF.

8. **Pipeline PNG status:** older PNG previews are retained as inactive historical assets; the manuscript includes the supplied vector PDF.

9. **Vector confirmation:** PDF text remains embedded as CID TrueType Liberation Sans text, and the diagram remains composed of vector text and shapes.

10. **TikZ fallback inclusion:** `figures/pipeline_overview_submission.tex` remains available as historical source but is not referenced by `main.tex`. The manuscript includes `figures/federated_mri_pipeline.pdf`.

11. **Pipeline labels and values:**
    - Stage 1: 7,200 MRI images; duplicate grouping; 6,116 image clusters; cluster-aware split; Train 5,040; Validation 720; Test 1,440; four balanced classes; exact hash, pixel duplicate, and same-class near-duplicate.
    - Stage 2: ResNet50, DenseNet121, EfficientNet-B0, EfficientNet-B3, and ConvNeXt-Tiny; selected EfficientNet-B0; best validation macro-F1 displayed as 0.9903 from source value 0.990262.
    - Stage 3: Server; Client 1 through Client 4; four virtual clients; IID; 70/30 Non-IID; FedAvg; FedProx; 20 rounds; three local epochs.
    - Stage 4: seeds 42, 123, and 2025; Non-IID accuracy; EfficientNet-B0 97.25 ± 0.51%; ResNet50 97.18 ± 0.28%; 82.8% lower full-state payload; 2,474.84 MiB versus 14,385.65 MiB per 20-round run.
    - Summary strip: “Comparable three-seed performance with 82.8% lower communication payload.”
    - Visual and text checks confirmed all four clients, all required values, and the absence of old headline values and identity text.

12. **Centralized screening sources:** `centralized_benchmark_results/model_comparison_val.csv`, `centralized_benchmark_results/CENTRALIZED_BENCHMARK_SUMMARY.md`, and `centralized_benchmark_results/config.json`.

    **Split-assignment provenance:** the dated generation record was recovered and checked against the preserved manifests. The split used 200 trials; trial $q$ used seed $42+1009q$, ordered clusters by decreasing size with seeded random tie-breaking, greedily minimized normalized class-wise and total-size target deviations with overshoot penalties, and retained the lowest-objective assignment. The selected assignment reached the exact balanced counts without dividing a cluster.

13. **Screening table values:**
    - ResNet50, layer3+layer4+fc, LR 0.0001, 23.516 M parameters, validation macro-F1 0.975004.
    - DenseNet121, all parameters, LR 0.0003, 6.958 M parameters, validation macro-F1 0.977831.
    - EfficientNet-B0, all parameters, LR 0.0003, 4.013 M parameters, validation macro-F1 0.990262.
    - EfficientNet-B3, all parameters, LR 0.0001, 10.702 M parameters, validation macro-F1 0.984731.
    - ConvNeXt-Tiny, all parameters, LR 0.0001, 27.823 M parameters, validation macro-F1 0.988885.

14. **Multi-seed result sources:** `reproducibility/summary/multiseed_results.json`, `.csv`, and `.md`; `reproducibility/statistics/paired_statistics.json`, `.csv`, and `.md`; and the six model-seed `result.json` files under `reproducibility/results/`.

15. **Primary three-seed values:** EfficientNet-B0: 97.25 ± 0.51% accuracy, 97.25 ± 0.51% macro-F1, 15.47 MiB state, and 2,474.84 MiB payload per run. ResNet50: 97.18 ± 0.28% accuracy, 97.18 ± 0.28% macro-F1, 89.91 MiB state, and 14,385.65 MiB payload per run. Mean paired accuracy and macro-F1 differences are each 0.07 percentage points with sample SD 0.32 points. Payload reduction is approximately 82.8%.

16. **Statistical values included:** seed 42 accuracy difference 0.42 points, CI −0.42 to 1.25, macro-F1 difference 0.41, CI −0.40 to 1.23, raw McNemar p=0.405032, Holm p=1.000; seed 123 accuracy and macro-F1 differences −0.21, CIs −1.11 to 0.69, raw p=0.760792, Holm p=1.000; seed 2025 accuracy difference 0.00, CI −0.69 to 0.69, macro-F1 approximately 0.00, CI −0.69 to 0.68, raw p=1.000, Holm p=1.000.

17. **Convergence histories used:**
    - `reproducibility/results/fedavg_efficientnet_b0_noniid_seed42/metrics_per_round_fedavg_efficientnet_b0_noniid_seed42.csv`
    - `reproducibility/results/fedavg_efficientnet_b0_noniid_seed123/metrics_per_round_fedavg_efficientnet_b0_noniid_seed123.csv`
    - `reproducibility/results/fedavg_efficientnet_b0_noniid_seed2025/metrics_per_round_fedavg_efficientnet_b0_noniid_seed2025.csv`
    - `reproducibility/results/fedavg_resnet50_noniid_seed42/metrics_per_round_fedavg_resnet50_noniid_seed42.csv`
    - `reproducibility/results/fedavg_resnet50_noniid_seed123/metrics_per_round_fedavg_resnet50_noniid_seed123.csv`
    - `reproducibility/results/fedavg_resnet50_noniid_seed2025/metrics_per_round_fedavg_resnet50_noniid_seed2025.csv`

18. **History validation:** every history contains each communication round 1 through 20 exactly once. The plot uses unrounded values and sample SD with `ddof=1`.

19. **Communication payload values:** EfficientNet-B0 2,474.84 MiB; ResNet50 14,385.65 MiB; reduction approximately 82.8%.

20. **Per-run confirmation:** every primary table, figure, caption, abstract sentence, discussion statement, and table note reports payload per 20-round federated run. Values were not multiplied by the three seeds.

    **Protocol and reproducibility disclosure:** the manuscript now states that both FL backbones deliberately shared AdamW at learning rate 0.0003 and weight decay 0.0001, that no backbone-specific FL tuning was performed, and that centralized optima were not transferred across the different optimization/fine-tuning regimes. It also records the RTX 5070 Ti, Python 3.14.6, PyTorch 2.12.1, torchvision 0.27.1, CUDA 13.0, and cuDNN 9.2 environment for the seed-123 and seed-2025 runs; the original seed-42 environment is explicitly identified as unavailable.

    **Contribution framing:** the Abstract, Introduction, contribution bullets, Related Work, and Conclusion consistently frame the paper around a duplicate-controlled, matched, statistically rigorous evaluation protocol. The manuscript does not claim a new FL algorithm, universal leakage elimination, statistical equivalence, or accuracy superiority.

21. **Final citation count:** 22 bibliography entries; all 22 are cited.

22. **Undefined citation count:** 0.

23. **Undefined reference count:** 0.

24. **Underfull warning count:** 2 vbox warnings; no visible layout defect was found.

25. **Overfull warning count:** 0.

26. **Margin inspection:** no text, table, figure, caption, arrow, or bibliography line visibly crosses a margin. All six pages were rendered at 160 DPI and inspected.

27. **Anonymity scan:** no specified author names, institution names, student identifiers, email addresses, affiliation text, repository links, usernames, acknowledgments, or local filesystem paths were found in active source, figure text, SVG, extracted PDF text, comments, captions, or build notes.

28. **PDF metadata:** title is correct; Author, Creator, Producer, Subject, and Keywords are blank; no metadata stream is present.

29. **SVG metadata:** title is “Experimental workflow,” description is generic, creator is blank, and no username, local path, email, repository, or machine identifier is present.

30. **Outdated-claim scan:** the previous paper title and prohibited superiority, equivalence, privacy, regulatory, and clinical-deployment claims are absent. No old single-seed primary comparison appears.

31. **Units:** all displayed payload and state sizes use MiB; no decimal-megabyte unit appears.

32. **Old primary comparison:** absent. The remaining 97.64% value is confined to the clearly labeled single-seed FedProx supporting row with seed field `1 (42)` and an explanatory table note.

33. **Reference-audit outcome:** the confirmed CFL-SparseMed author error was corrected to G. Habib, A. Bhardwaj, R. Sharma, S. A. Banday, and I. A. Malik. The Federated Cross Learning entry was also corrected by adding P. Yan and the official MIDL/PMLR metadata. The Kaggle entry now identifies Version 2, its URL, and the exact access date. The Saraswat and Radhika venue metadata should still receive a final source-level check; the verified Bentemessek item is a bachelor thesis rather than a peer-reviewed paper.

34. **Changed source and support files:**
    - `paper_anonymous_submission/main.tex`
    - `paper_anonymous_submission/references.bib`
    - `paper_anonymous_submission/README_BUILD.md`
    - `paper_anonymous_submission/figures/federated_mri_pipeline.pdf`
    - `paper_anonymous_submission/FINAL_SUBMISSION_REPORT.md`

35. **Generated or regenerated files:**
    - `paper_anonymous_submission/main.pdf`, `main.aux`, `main.bbl`, `main.blg`, `main.log`, and `main.out`
    - `paper_anonymous_submission/final_pdf_text.txt`
    - six temporary 160-DPI diagnostic page renders under the system temporary directory; these are not submission artifacts.

36. **Training:** no training ran.

37. **Inference:** no checkpoint inference ran.

38. **Dataset files:** no dataset file was modified.

39. **Manifests:** no manifest was modified; train, validation, and test manifests passed size, class, hash, and overlap validation.

40. **Client partitions:** no partition was modified; all three shared 70/30 Non-IID partitions passed recorded-hash and assignment validation.

41. **Checkpoints:** no checkpoint was modified; state dictionaries were read only for size verification.

42. **Prediction files:** no prediction file was modified; all six prediction sets passed sample-ID, label, probability, and metric recalculation checks.

43. **Source result files:** no source result JSON, CSV, or Markdown file was modified.

44. **Result-file hash verification:** protected scientific artifacts matched their pre-edit aggregate SHA-256 values after revision: `cluster_split_outputs` `89d50eef3e8201170a16fd0ff4bef16413e7d80913b541ff9ee40b277cc18824`; `centralized_benchmark_results` `34f60c0b90139d662546572061479815b2533d37bdf85a9141cce07443e53ea5`; `federated_results_20rounds` `ed5ae54158f078f0b32f65ae690e19ea1c5ba6377e77b457a17a2abab14e1af0`; `reproducibility/results` `d57c23b37bb5382189693d0722b4ebc13e7c5802fd997dbdcf1107eab6937651`; `reproducibility/statistics` `e615d944ac7e5b5bb086486988265dff9d7be3eedb473c36cf501b652fd7b92c`; `reproducibility/summary` `44d51b16674ca249b5e53bf2eb74249d7557447c49f199b27cb4c588fb9c01b2`; and `reproducibility/client_partitions` `ecb67357dd22b0525214879ae54b68c1e7baa60892b743a24c7ff8a8439e8912`. The bibliography was intentionally corrected and is not a protected result source.

45. **Submission readiness:** the supplied vector pipeline is included, the historical TikZ pipeline is excluded, boxed one-column tables fit, the restored confusion matrix is present, the paper is six pages, the title and metadata are correct, citations resolve, anonymity passes, numerical values match saved outputs, figures are readable, protected files are unchanged, and no training or inference ran.

FINAL STATUS: READY FOR SUBMISSION
