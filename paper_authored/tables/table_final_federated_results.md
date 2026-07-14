| Method | Backbone | Partition | Best round | Accuracy (%) | Macro-F1 (%) | Parameters (M) | State dict (MB) | Total communication (MB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FedAvg | EfficientNet-B0 | IID | 20 | 97.92 | 97.92 | 4.01 | 15.47 | 2474.84 |
| FedAvg | EfficientNet-B0 | Non-IID | 20 | 97.64 | 97.64 | 4.01 | 15.47 | 2474.84 |
| FedProx | EfficientNet-B0 | Non-IID | 20 | 97.64 | 97.64 | 4.01 | 15.47 | 2474.84 |
| FedAvg | ResNet50 | Non-IID | 15 | 97.22 | 97.23 | 23.52 | 89.91 | 14385.65 |
