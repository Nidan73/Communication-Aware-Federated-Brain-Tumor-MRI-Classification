Here is a Related Work writing map tailored to your paper, utilizing the sources provided in the NotebookLM workspace and adhering to your specific framing constraints.

### 2.1 Federated Learning for Medical Imaging

**1. Selected References:**
*   [Rieke2020]: Rieke et al. (2020) - *The future of digital health with federated learning*
*   [Sheller2020]: Sheller et al. (2020) - *Federated learning in medicine: facilitating multi-institutional collaborations...*
*   [Sharma2025]: Sharma et al. (2025) - *A privacy-preserved horizontal federated learning for malignant glioma tumour detection...*
*   [Babar2024]: Babar et al. (2024) - *Investigating the impact of data heterogeneity on the performance of federated learning...*
*   [Ghanta2025]: Ghanta et al. (2025) - *Federated proximal learning with data augmentation for brain tumor classification...*
*   [Mastoi2025]: Mastoi et al. (2025) - *Explainable AI in medical imaging: an interpretable and collaborative federated learning model...*
*   [Xu2023]: Xu et al. (2023) - *Federated Cross Learning for Medical Image Segmentation*

**2. What this group contributes:** 
These works establish the necessity and viability of Federated Learning (FL) in healthcare to bypass strict privacy regulations (e.g., HIPAA, GDPR) without centralizing sensitive datasets. They explore foundational aggregation techniques like FedAvg and introduce optimizations such as FedProx to handle the non-independent and identically distributed (Non-IID) label skew and feature shifts commonly found in multi-institutional medical imaging.

**3. The gap your paper addresses:** 
While existing literature validates FL for medical imaging using varying levels of Non-IID constraints, many studies deploy models without rigorously pre-screening lightweight backbones or systematically comparing FedAvg and FedProx under identical, highly controlled simulated communication rounds.

**4. Draft Paragraph:**
Federated learning has gained significant traction in digital health by enabling collaborative model training across institutions without the need to centralize sensitive patient data [Rieke2020, Sheller2020]. Recent studies have successfully applied FL to oncological imaging tasks, demonstrating that decentralized protocols can offer competitive classification performance [Mastoi2025, Sharma2025]. However, real-world clinical distributions frequently suffer from statistical heterogeneity. This has prompted the adoption of advanced optimization strategies such as FedProx, which introduces a proximal term to mitigate client drift under non-IID conditions [Ghanta2025, Babar2024, Xu2023]. Building upon these algorithmic advancements, our study integrates a centralized backbone screening phase prior to federated deployment, systematically evaluating FedAvg and FedProx optimization over 20 communication rounds across four simulated clients.

---

### 2.2 Brain Tumor MRI Classification

**1. Selected References:**
*   [Nickparvar2021]: Nickparvar (2021) - *Brain Tumor MRI Dataset*
*   [Khatoon2026]: Khatoon & Patel (2026) - *A ResNet50-Based Deep Learning Framework with Optimized MRI Preprocessing*
*   [Bentemessek2025]: Bentemessek (2025) - *Brain Tumor Classification using Deep Learning: Custom CNN vs. ResNet50*
*   [Ghosh2026]: Ghosh (2026) - *Privacy-Preserving Brain Tumor Classification Using Federated Learning with ResNet50...*
*   [Saraswat2026]: Saraswat & Pandey (2026) - *Hybrid Cnn–Transformer Federated Learning for Brain Tumor Segmentation...*
*   [Melo2026]: Melo et al. (2026) - *Exploiting Test-Time Augmentation in Federated Learning for Brain Tumor MRI Classification*
*   [Siam2026]: Siam & Khan (2026) - *Brain tumor MRI image for Fedarated Learning*

**2. What this group contributes:**
This cluster benchmarks the application of deep learning for multi-class brain tumor classification (glioma, meningioma, pituitary, no tumor). The literature heavily utilizes standard architectures like ResNet50 as a performance baseline. Recently, works like Melo et al. and Siam & Khan have highlighted the critical risk of data leakage in public datasets, advocating for rigorous duplicate removal (e.g., via SHA-256) and cluster-aware/patient-wise dataset partitioning. 

**3. The gap your paper addresses:** 
Many contemporary classification studies rely on standard random splits and computationally heavy centralized models (like ResNet50) that can suffer from optimistic bias due to overlapping slice data. Your paper addresses this by implementing a strict cluster-aware, duplicate-controlled splitting methodology to guarantee reliable evaluation metrics when deploying lightweight models.

**4. Draft Paragraph:**
Deep learning has substantially automated brain tumor MRI classification, with numerous studies benchmarking standard architectures—particularly ResNet50—on public datasets to classify gliomas, meningiomas, and pituitary tumors [Khatoon2026, Bentemessek2025, Nickparvar2021]. As the field transitions toward decentralized clinical applications, frameworks have emerged that integrate these established architectures into federated networks [Ghosh2026, Saraswat2026]. However, recent literature indicates that standard evaluation protocols often suffer from data leakage when consecutive MRI slices from the same patient are distributed across training and testing sets [Melo2026, Siam2026]. To yield reliable diagnostic metrics, dataset partitioning must stringently eliminate duplicates. This work addresses this requirement by implementing a cluster-aware, duplicate-controlled evaluation strategy to ensure a rigorous and unbiased assessment of the underlying classification models.

---

### 2.3 Communication and Backbone Efficiency

**1. Selected References:**
*   [McMahan2017]: McMahan et al. (2017) - *Communication-Efficient Learning of Deep Networks from Decentralized Data*
*   [Radhika2026]: Radhika et al. (2026) - *Communication-Efficient Federated Learning (CEFL) for CT Image Classification...*
*   [Aniket2025]: Aniket (2025) - *CFL-SparseMed: Communication-Efficient Federated Learning for Medical Imaging...*
*   [Zhou2022]: Zhou (2022) - *Communication-Efficient Federated Learning for Multi-Institutional Medical Image Classification*
*   [Abubakr2026]: Abubakr et al. (2026) - *Explainable Federated Attention-Based Deep Learning for Alzheimer's Disease Detection*
*   [Tan2019]: Tan & Le (2019) - *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*

**2. What this group contributes:** 
These references address the network latency and bandwidth constraints inherent in decentralized learning. They explore techniques to minimize the payload sent over wireless healthcare networks, such as gradient sparsification, parameter limits, and the deployment of inherently lightweight backbone architectures like EfficientNet-B0 to reduce the total multi-round communication cost. 

**3. The gap your paper addresses:** 
While existing research proposes various mathematical compression techniques to reduce payload size, there is a lack of empirical side-by-side accounting of full-state communication costs (upload and download) comparing standard heavy baselines (ResNet50) directly against screened lightweight architectures (EfficientNet-B0) in a fixed-round federated environment.

**4. Draft Paragraph:**
The iterative exchange of parameters in federated networks introduces substantial communication overhead, representing a major bottleneck for bandwidth-constrained wireless healthcare settings [McMahan2017]. To mitigate this, researchers have explored algorithmic gradient compression and sparsification strategies [Radhika2026, Aniket2025, Zhou2022]. A complementary approach relies on deploying inherently parameter-efficient model backbones. For instance, [Abubakr2026] and [Tan2019] demonstrated the efficacy of EfficientNet-B0 in achieving competitive diagnostic accuracy while maintaining a highly reduced parameter footprint. While previous studies frequently evaluate optimization algorithms and backbone architectures independently, this work provides a comprehensive full-state communication accounting analysis. Specifically, the network efficiency and classification robustness of EfficientNet-B0 are directly compared against the conventional ResNet50 architecture under an identical 20-round simulated federated learning environment.