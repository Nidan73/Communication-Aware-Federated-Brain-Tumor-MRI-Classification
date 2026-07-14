Based on the provided documents, I have identified **23 strong, highly relevant sources** that perfectly map to your required categories. Dedicated foundational papers for DenseNet and ConvNeXt were not present in the uploaded texts, but they are benchmarked inside the included comparative studies. 

Here is the source inventory for your paper:

### Core Federated Learning & Optimization (FedAvg / FedProx)
**1. Communication-Efficient Learning of Deep Networks from Decentralized Data**
*   **Authors:** B. McMahan, E. Moore, D. Ramage, S. Hampson, B. A. y Arcas
*   **Year:** 2017
*   **Venue:** AISTATS
*   **DOI/URL:** arXiv:1602.05629
*   **Reason:** The foundational paper introducing the FedAvg algorithm.
*   **Suggested location:** Introduction / Related Work: Core federated learning
*   **Confidence level:** High

**2. Federated Optimization in Heterogeneous Networks (FedProx)**
*   **Authors:** T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith
*   **Year:** 2020
*   **Venue:** MLSys
*   **DOI/URL:** Not available in provided source
*   **Reason:** The seminal paper introducing the FedProx algorithm to handle Non-IID data distributions.
*   **Suggested location:** Introduction / Methodology (Federated Optimization)
*   **Confidence level:** High

**3. Federated proximal learning with data augmentation for brain tumor classification under heterogeneous data distributions**
*   **Authors:** Swetha Ghanta, Venkata Sumanth Siddareddy, Prasanthi Boyapati, Sujit Biswas, Gandharba Swain, Ashok K. Pradhan
*   **Year:** 2025
*   **Venue:** PeerJ Computer Science
*   **DOI/URL:** 10.7717/peerj-cs.3396
*   **Reason:** Directly mirrors your work by applying FedProx to handle label-skewed Non-IID brain tumor MRI data. 
*   **Suggested location:** Related Work / Methodology
*   **Confidence level:** High

### Federated Learning in Medical Imaging & Brain Tumor Classification
**4. The future of digital health with federated learning**
*   **Authors:** N. Rieke, J. Hancox, W. Li, et al.
*   **Year:** 2020
*   **Venue:** NPJ Digital Medicine
*   **DOI/URL:** 10.1038/s41746-020-00323-1
*   **Reason:** Provides a comprehensive overview of the necessity and challenges of FL in digital health.
*   **Suggested location:** Introduction
*   **Confidence level:** High

**5. Privacy-Preserving Brain Tumor Classification Using Federated Learning with ResNet50 and Grad-CAM**
*   **Authors:** Sowrov Ghosh
*   **Year:** 2026
*   **Venue:** ICCA 2026
*   **DOI/URL:** Not available in provided source
*   **Reason:** A highly related study implementing FedAvg, ResNet50, and 70/30 Non-IID splits on the Nickparvar dataset.
*   **Suggested location:** Related Work / Results
*   **Confidence level:** High

**6. A privacy-preserved horizontal federated learning for malignant glioma tumour detection using distributed data-silos**
*   **Authors:** Shagun Sharma, Kalpna Guleria, Ayush Dogra, Deepali Gupta, Sapna Juneja, Swati Kumari, Ali Nauman
*   **Year:** 2025
*   **Venue:** PLOS ONE
*   **DOI/URL:** 10.1371/journal.pone.0316543
*   **Reason:** Evaluates horizontal FL with lightweight backbones (MobileNetV2) under IID and Non-IID settings.
*   **Suggested location:** Related Work: FL medical imaging
*   **Confidence level:** High

**7. Explainable AI in medical imaging: an interpretable and collaborative federated learning model for brain tumor classification**
*   **Authors:** Qurat-ul-ain Mastoi, Shahid Latif, Sarfraz Brohi, Jawad Ahmad, Abdulmajeed Alqhatani, Mohammed S. Alshehri, Alanoud Al Mazroa, Rahmat Ullah
*   **Year:** 2025
*   **Venue:** Frontiers in Oncology
*   **DOI/URL:** 10.3389/fonc.2025.1535478
*   **Reason:** Validates collaborative FL using a heavy backbone (GoogLeNet) for four-class brain tumor classification.
*   **Suggested location:** Related Work: brain tumor MRI classification
*   **Confidence level:** High

**8. Hybrid Cnn–Transformer Federated Learning for Brain Tumor Segmentation Across Multi-Center Mri Datasets**
*   **Authors:** Anil Saraswat, Nageshwar Nath Pandey
*   **Year:** 2026
*   **Venue:** Int J Drug Deliv Technol
*   **DOI/URL:** 10.25258/ijddt.16.30s.28
*   **Reason:** Discusses multi-center data privacy, Non-IID conditions, and compares CNN performance against transformers in FL.
*   **Suggested location:** Related Work
*   **Confidence level:** Medium

**9. TwinSegNet: A Digital Twin-Enabled Federated Learning Framework for Brain Tumor Analysis**
*   **Authors:** Almustapha A. Wakili, Adamu Hussaini, Abubakar A. Musa, Woosub Jung, Wei Yu
*   **Year:** 2025
*   **Venue:** arXiv
*   **DOI/URL:** arXiv:2512.17488v1
*   **Reason:** Addresses Non-IID distributions in multi-institutional brain MRI tasks with lightweight adaptation.
*   **Suggested location:** Related Work
*   **Confidence level:** Medium

### Model Backbones (ResNet, EfficientNet & Centralized Baselines)
**10. Deep Residual Learning for Image Recognition**
*   **Authors:** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
*   **Year:** 2016
*   **Venue:** CVPR
*   **DOI/URL:** 10.1109/CVPR.2016.90
*   **Reason:** Foundational citation for the ResNet50 backbone architecture.
*   **Suggested location:** Methodology
*   **Confidence level:** High

**11. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**
*   **Authors:** Mingxing Tan, Quoc V. Le
*   **Year:** 2019
*   **Venue:** ICML
*   **DOI/URL:** arXiv:1905.11946
*   **Reason:** Foundational citation for the EfficientNet-B0 backbone architecture.
*   **Suggested location:** Methodology
*   **Confidence level:** High

**12. Brain Tumor Classification using Deep Learning: Custom CNN vs. ResNet50**
*   **Authors:** Rayen Bentemessek
*   **Year:** 2025
*   **Venue:** CCT College Dublin
*   **DOI/URL:** 10.63227/652.199.42
*   **Reason:** Establishes ResNet50 as a strong centralized baseline for tumor detection.
*   **Suggested location:** Methodology / Baselines
*   **Confidence level:** Medium

**13. A ResNet50-Based Deep Learning Framework with Optimized MRI Preprocessing**
*   **Authors:** Sumaiya Khatoon, Deepshikha Patel
*   **Year:** 2026
*   **Venue:** IJERT
*   **DOI/URL:** 10.17577/IJERTV14IS120635
*   **Reason:** Further validates ResNet50 on the 4-class Kaggle dataset, serving as a comparison point for your screening phase.
*   **Suggested location:** Related Work
*   **Confidence level:** Medium

**14. Explainable Federated Attention-Based Deep Learning for Alzheimer's Disease Detection**
*   **Authors:** Noor Abubakr, Mustafa Kara, Hasan Hüseyin Balık
*   **Year:** 2026
*   **Venue:** Applied Sciences
*   **DOI/URL:** 10.3390/app16104648
*   **Reason:** Provides recent literature justification for utilizing EfficientNet-B0 as a lightweight, communication-efficient backbone in medical FL.
*   **Suggested location:** Methodology / Architecture Justification
*   **Confidence level:** High

### Communication Efficiency & Accounting
**15. Communication-Efficient Federated Learning (CEFL) for CT Image Classification in Bandwidth-Constrained Wireless Healthcare Networks**
*   **Authors:** Radhika K R, H. Nagesh Shenoy, Vinay T R, Pooja H, Dolly Sharma, Priyanka R, Sumit Gupta
*   **Year:** 2026
*   **Venue:** Int J Drug Deliv Technol
*   **DOI/URL:** 10.25258/ijddt.16.13s.17
*   **Reason:** Extensively details the tracking of communication cost (upload/download sizes) and latency tradeoffs between FedAvg and FedProx.
*   **Suggested location:** Related Work: communication efficiency
*   **Confidence level:** High

**16. CFL-SparseMed: Communication-Efficient Federated Learning for Medical Imaging with Top-k Sparse Updates**
*   **Authors:** Aniket (Full author list not extracted in text, usually cited by title)
*   **Year:** 2025
*   **Venue:** arXiv
*   **DOI/URL:** arXiv:2510.24776v1
*   **Reason:** Reinforces the necessity of reducing communication overhead in FL for heavy medical models.
*   **Suggested location:** Related Work
*   **Confidence level:** Medium

**17. Communication-Efficient Federated Learning for Multi-Institutional Medical Image Classification**
*   **Authors:** Shuang Zhou
*   **Year:** 2022
*   **Venue:** Proc SPIE Int Soc Opt Eng
*   **DOI/URL:** 10.1117/12.2611654
*   **Reason:** Early framework establishing metrics for parameter uploading limits in Non-IID healthcare networks.
*   **Suggested location:** Related Work
*   **Confidence level:** Medium

**18. Federated learning framework for medical image analysis with perspective-aware contrastive and mixture of experts**
*   **Authors:** Shradhanjali Das, K. Hemalatha
*   **Year:** 2026
*   **Venue:** Frontiers in Artificial Intelligence
*   **DOI/URL:** 10.3389/frai.2026.1807248
*   **Reason:** Contains rigorous benchmarks on FLOPs and communication costs in distributed medical networks.
*   **Suggested location:** Results / Discussion (Comparing overhead)
*   **Confidence level:** High

### Data Leakage, Dataset Splitting, and Limitations
**19. Exploiting Test-Time Augmentation in Federated Learning for Brain Tumor MRI Classification**
*   **Authors:** Thamara Leandra de Deus Melo, Rodrigo Moreira, Larissa Ferreira Rodrigues Moreira, André R. Backes
*   **Year:** 2026
*   **Venue:** arXiv
*   **DOI/URL:** arXiv:2601.12671v1
*   **Reason:** Explicitly documents the removal of exact duplicate images via SHA-256 to prevent data leakage in federated MRI partitioning.
*   **Suggested location:** Methodology (Duplicate-controlled evaluation)
*   **Confidence level:** High

**20. Brain tumor MRI image for Fedarated Learning**
*   **Authors:** Masrafe Bin Hannan Siam, Rayhan Khan
*   **Year:** 2026
*   **Venue:** Mendeley Data
*   **DOI/URL:** 10.17632/x69nzw9zzw.2
*   **Reason:** Explains the necessity of patient-wise/cluster-aware dataset splitting into exactly 4 clients to prevent data leakage.
*   **Suggested location:** Methodology
*   **Confidence level:** High

**21. MedFL-Stress: A Systematic Robustness Evaluation of Federated Brain Tumor Segmentation under Cross-Hospital MRI Appearance Shift**
*   **Authors:** Kiran Naseer, Anwer Butt
*   **Year:** 2026
*   **Venue:** arXiv
*   **DOI/URL:** arXiv:2605.09025v1
*   **Reason:** Highlights critical limitations in FL medical AI, arguing that standard global evaluations obscure data leakage and worst-case client disparities.
*   **Suggested location:** Limitations / Discussion
*   **Confidence level:** High

**22. Federated Cross Learning for Medical Image Segmentation**
*   **Authors:** Xuanang Xu, Hannah H. Deng, Pingkun Yan
*   **Year:** 2023
*   **Venue:** MIDL 2023
*   **DOI/URL:** Not available in provided source
*   **Reason:** Provides theoretical analysis on why traditional model aggregation fails on non-IID MRI datasets.
*   **Suggested location:** Discussion / Limitations
*   **Confidence level:** Medium

### Datasets
**23. Brain Tumor MRI Dataset**
*   **Authors:** Masoud Nickparvar
*   **Year:** 2021
*   **Venue:** Kaggle
*   **DOI/URL:** 10.34740/KAGGLE/DSV/2645886
*   **Reason:** The official citation for the 4-class, 7,023 image target dataset.
*   **Suggested location:** Methodology (Dataset)
*   **Confidence level:** High