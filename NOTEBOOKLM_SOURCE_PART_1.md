1. Full title: Federated proximal learning with data augmentation for brain tumor classification under heterogeneous data distributions
2. Authors: Swetha Ghanta, Venkata Sumanth Siddareddy, Prasanthi Boyapati, Sujit Biswas, Gandharba Swain, Ashok K. Pradhan
3. Year: 2025
4. Venue: PeerJ Computer Science
5. DOI: 10.7717/peerj-cs.3396
6. Main topic: Addressing non-IID data distributions in federated learning for brain tumor classification.
7. Method/model used: FedProx, ResNet50, data augmentation.
8. Dataset used: Kaggle Brain Tumor MRI dataset (7,023 images).
9. Key result: The FedProx algorithm combined with ResNet50 achieved 97.71% accuracy on IID data and 87.19% on extreme non-IID scenarios across 4 simulated clients.
10. Why it is relevant: It perfectly aligns with your use of 4 virtual clients, ResNet50, and the FedProx algorithm to combat heterogeneous (non-IID) data distributions.
11. Where I should cite it: 
   - Related Work: FL medical imaging
   - Related Work: brain tumor MRI classification
   - Methodology

---

1. Full title: Privacy-Preserving Brain Tumor Classification Using Federated Learning with ResNet50 and Grad-CAM
2. Authors: Sowrov Ghosh (Additional authors not available in provided source)
3. Year: 2026
4. Venue: ICCA 2026
5. URL/DOI: Not available in provided source
6. Main topic: Decentralized privacy-preserving brain tumor classification with visual explainability.
7. Method/model used: FedAvg, ResNet50.
8. Dataset used: Nickparvar Brain Tumor MRI dataset (7,023 images; 4 classes).
9. Key result: Achieved 95.2% accuracy in IID settings and 94.1% in a 70/30 Non-IID setting. It also tracked cumulative communication costs, which reached ~1.7 GB over 5 rounds.
10. Why it is relevant: This paper exactly mirrors your 70/30 label-skew Non-IID partition, the ResNet50 backbone, FedAvg optimization, and the practice of tracking communication costs (model weight upload/download) over training rounds.
11. Where I should cite it: 
   - Related Work: brain tumor MRI classification
   - Methodology
   - Results/Discussion

---

1. Full title: Exploiting Test-Time Augmentation in Federated Learning for Brain Tumor MRI Classification
2. Authors: Thamara Leandra de Deus Melo, Rodrigo Moreira, Larissa Ferreira Rodrigues Moreira, André R. Backes
3. Year: 2026
4. Venue: arXiv / VISAPP
5. arXiv ID: 2601.12671v1
6. Main topic: Evaluating the impact of image preprocessing and inference augmentation on FL classification stability.
7. Method/model used: FedAvg, ResNet-18, SHA-256 exact duplicate removal.
8. Dataset used: Kaggle Brain Tumor MRI dataset.
9. Key result: The authors identified and removed 194 exact duplicate images using SHA-256 hashing before splitting the dataset to prevent data leakage and optimistic bias. 
10. Why it is relevant: Directly supports your methodology of applying duplicate-controlled/cluster-aware dataset splitting to ensure rigorous and unbiased federated partitions.
11. Where I should cite it: 
   - Methodology
   - Limitations

---

1. Full title: Explainable Federated Attention-Based Deep Learning for Alzheimer's Disease Detection
2. Authors: Noor Abubakr, Mustafa Kara, Hasan Hüseyin Balık
3. Year: 2026
4. Venue: Applied Sciences
5. DOI: 10.3390/app16104648
6. Main topic: Privacy-preserving and explainable deep learning framework for MRI classification.
7. Method/model used: FedAvg, EfficientNetB0 + Convolutional Block Attention Module (CBAM).
8. Dataset used: Publicly available Alzheimer's MRI datasets.
9. Key result: Demonstrated that an EfficientNet-B0 architecture acts as a highly effective, lightweight backbone for federated learning, achieving 94.66% accuracy in a federated setting with reduced parameter overhead.
10. Why it is relevant: Provides strong architectural justification for selecting EfficientNet-B0 as a communication-efficient alternative to heavier models like ResNet50 in decentralized medical imaging.
11. Where I should cite it: 
   - Related Work: communication efficiency
   - Results/Discussion

---

1. Full title: Communication-Efficient Federated Learning (CEFL) for CT Image Classification in Bandwidth-Constrained Wireless Healthcare Networks
2. Authors: Radhika K R, H. Nagesh Shenoy, Vinay T R, Pooja H, Dolly Sharma, Priyanka R, Sumit Gupta
3. Year: 2026
4. Venue: Int J Drug Deliv Technol
5. DOI: 10.25258/ijddt.16.13s.17
6. Main topic: Reducing communication overhead in bandwidth-constrained medical federated networks.
7. Method/model used: FedAvg, FedProx, Gradient compression, Model quantization.
8. Dataset used: LIDC-IDRI CT dataset.
9. Key result: Exposed the latency versus accuracy tradeoff between FedAvg and FedProx, proving that optimizing communication can reduce overhead by 40-60% while retaining ~90% accuracy.
10. Why it is relevant: Grounds your communication accounting strategy. It provides literature backing for comparing the communication efficiency and latency/tradeoffs of FedAvg versus FedProx.
11. Where I should cite it: 
   - Related Work: communication efficiency
   - Results/Discussion

---

1. Full title: Brain tumor MRI image for Fedarated Learning
2. Authors: Masrafe Bin Hannan Siam, Rayhan Khan
3. Year: 2026
4. Venue: Mendeley Data
5. DOI: 10.17632/x69nzw9zzw.2
6. Main topic: Structuring brain tumor MRI data specifically for privacy-preserving federated simulations.
7. Method/model used: Synchronous FedAvg, 4-client partitioning, patient-wise dataset splitting.
8. Dataset used: 10,417 brain MRI slices (4 clinical categories).
9. Key result: Emphasized that data must be distributed into 4 distinct client partitions and strictly divided using an 80/10/10 patient-wise split to prevent data leakage during FL simulations.
10. Why it is relevant: Strongly validates your experimental setup of using 4 virtual clients and reinforces the necessity of cluster-aware/patient-wise dataset splitting to prevent data leakage across clients.
11. Where I should cite it: 
   - Methodology

---

1. Full title: Brain Tumor Classification using Deep Learning: Custom CNN vs. ResNet50
2. Authors: Rayen Bentemessek
3. Year: 2025
4. Venue: CCT College Dublin
5. DOI: 10.63227/652.199.42
6. Main topic: Comparative analysis of centralized convolutional neural networks for tumor detection.
7. Method/model used: Custom CNN, ResNet50.
8. Dataset used: Masoud Nickparvar Kaggle Dataset (7,023 images).
9. Key result: Highlighted ResNet50's strong feature extraction capability but warned against the risks of data leakage in brain tumor dataset curation.
10. Why it is relevant: Serves as an excellent reference for your centralized backbone screening phase, specifically validating ResNet50 as a standard comparative baseline before transitioning to federated testing.
11. Where I should cite it: 
   - Introduction
   - Related Work: brain tumor MRI classification

---

1. Full title: A privacy-preserved horizontal federated learning for malignant glioma tumour detection using distributed data-silos
2. Authors: Shagun Sharma, Kalpna Guleria, Ayush Dogra, Deepali Gupta, Sapna Juneja, Swati Kumari, Ali Nauman
3. Year: 2025
4. Venue: PLOS ONE
5. DOI: 10.1371/journal.pone.0316543
6. Main topic: Horizontal federated learning for detecting glioma utilizing distributed silos.
7. Method/model used: FedAvg, MobileNetV2.
8. Dataset used: Kaggle Brain Tumor MRI dataset (glioma vs. no tumor).
9. Key result: Successfully tested IID and non-IID distributions using a lightweight pre-trained model (MobileNetV2), achieving over 99% accuracy while explicitly addressing computational/communication overhead constraints.
10. Why it is relevant: Acts as a foundational study justifying the need to screen various backbones (like your centralized screening of DenseNet, ConvNeXt, EfficientNet) to find lightweight alternatives for FL that excel in both IID and Non-IID environments.
11. Where I should cite it: 
   - Introduction
   - Related Work: FL medical imaging