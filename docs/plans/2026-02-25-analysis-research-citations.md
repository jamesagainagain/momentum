# Research & citations for temporal topic analysis pipeline

This document cites sources that support the design choices in the [clean-pipeline](2026-02-25-clean-pipeline-recluster.md) and [temporal-topic-analysis design](2026-02-21-temporal-topic-analysis-design.md) plans.

---

## Clustering (HDBSCAN)

- **Decision:** Use HDBSCAN for topic discovery on embeddings without fixing the number of clusters.
- **Source:** Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. *PAKDD 2013*, Springer, pp. 160–172. DOI: [10.1007/978-3-642-37456-2_14](https://doi.org/10.1007/978-3-642-37456-2_14).
- **Relevance:** Hierarchical density estimates and stability-based extraction allow discovery of natural cluster boundaries without specifying K; the method is robust to varying density and assigns noise points to cluster -1.

**Optional:** Wang et al. (2021). Improving the Performance of HDBSCAN on Short Text Clustering by Using Word Embedding and UMAP. *IEEE*. DOI: [10.1109/ICTAI52525.2021.00100](https://ieeexplore.ieee.org/document/9640285) — supports combining HDBSCAN with embeddings and UMAP for short-text topic clustering.

---

## Visualization (UMAP)

- **Decision:** Use UMAP for 2D projection of embeddings for cluster visualization.
- **Source:** McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426*; also *Journal of Open Source Software*, 3(29), 861. [https://arxiv.org/abs/1802.03426](https://arxiv.org/abs/1802.03426)
- **Relevance:** Preserves both local and global structure better than t-SNE for visualization; scalable and applicable to high-dimensional embeddings.

---

## Temporal topic analysis and evolution

- **Decision:** Track topics over time windows (e.g. monthly/weekly), and classify lifecycle states (emerging, trending, stable, declining, dormant).
- **Sources:**
  - Rudolph, M., Blei, D. (2019). Dynamic Embedded Topic Model. *arXiv:1907.05545*. [https://arxiv.org/abs/1907.05545](https://arxiv.org/abs/1907.05545) — models topics as time-varying vectors; smooth topic evolution over time.
  - Aligned Neural Topic Model (ANTM), arXiv:2302.01501 — time-aware features and sliding-window clustering for evolving topics in social media.
  - TM-LDA: efficient online modeling of latent topic transitions in social media. *ACM*, 2012. [https://dl.acm.org/doi/10.1145/2339530.2339552](https://dl.acm.org/doi/10.1145/2339530.2339552)
  - Chronotome: Real-Time Topic Modeling for Streaming Embedding Spaces. *arXiv:2509.01051* — real-time visualization and streaming clustering for temporal datasets.
- **Relevance:** Justifies time-windowed aggregation and lifecycle-style classification in social and temporal settings.

---

## Preprocessing (deduplication, author capping)

- **Decision:** Remove exact and near-duplicate posts; cap single-author share to limit spam/bot dominance before clustering.
- **Sources:**
  - Generative Deduplication For Social Media Data Selection. *ACL Findings EMNLP*, 2024. [https://aclanthology.org/2024.findings-emnlp.330/](https://aclanthology.org/2024.findings-emnlp.330/)
  - Scalable and Generalizable Social Bot Detection through Data Selection. *AAAI*. [https://aaai.org/papers/01096-scalable-and-generalizable-social-bot-detection-through-data-selection](https://aaai.org/papers/01096-scalable-and-generalizable-social-bot-detection-through-data-selection)
  - Cresci, S. et al. Social media bot detection with deep learning methods: a systematic review. *Neural Computing and Applications*, Springer, 2023. [https://link.springer.com/article/10.1007/s00521-023-08352-z](https://link.springer.com/article/10.1007/s00521-023-08352-z)
  - Unsupervised botnet detection (duplicate content, shortened URLs). *arXiv:1804.05232* — real-time detection of bot groups posting duplicate content.
- **Relevance:** Supports preprocessing before clustering to improve topic quality and reduce bias from bots and duplicate content; estimates of bot prevalence (e.g. 9–17% of accounts) motivate author capping and deduplication.

---

## Event / anomaly detection

- **Decision:** Detect spikes and drops via anomaly scores (e.g. z-score) for temporal events in cluster time series.
- **Sources:**
  - Blázquez-García, A. et al. (2020). A review on outlier/anomaly detection in time series data. *arXiv:2002.04236*. [https://arxiv.org/abs/2002.04236](https://arxiv.org/abs/2002.04236)
  - Schmidl, S. et al. Anomaly Detection in Time Series: A Comprehensive Evaluation. *TimeEval*. [https://timeeval.github.io/evaluation-paper/](https://timeeval.github.io/evaluation-paper/)
- **Relevance:** Justifies threshold-based event detection (e.g. anomaly_score > 2.0) for “spike”/“drop” events; statistical methods such as z-scores are well-established for point anomalies in time series.

---

## References (full list)

1. Campello, R.J.G.B., Moulavi, D., Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. In *Advances in Knowledge Discovery and Data Mining (PAKDD 2013)*, Springer, Berlin, Heidelberg, pp. 160–172. DOI: 10.1007/978-3-642-37456-2_14.

2. Wang et al. (2021). Improving the Performance of HDBSCAN on Short Text Clustering by Using Word Embedding and UMAP. *IEEE International Conference on Tools with Artificial Intelligence (ICTAI)*. DOI: 10.1109/ICTAI52525.2021.00100.

3. McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. *arXiv:1802.03426* [stat.ML]. *Journal of Open Source Software*, 3(29), 861.

4. Rudolph, M., Blei, D. (2019). Dynamic Embedded Topic Model. *arXiv:1907.05545*.

5. ANTM: An Aligned Neural Topic Model For Exploring Evolving Topics. *arXiv:2302.01501*.

6. Hong, L., Davison, B.D. (2012). TM-LDA: efficient online modeling of latent topic transitions in social media. *ACM KDD*.

7. Chronotome: Real-Time Topic Modeling for Streaming Embedding Spaces. *arXiv:2509.01051*.

8. Generative Deduplication For Social Media Data Selection. *ACL Findings of EMNLP*, 2024. ACL Anthology 2024.findings-emnlp.330.

9. Scalable and Generalizable Social Bot Detection through Data Selection. *AAAI*.

10. Cresci, S. et al. (2023). Social media bot detection with deep learning methods: a systematic review. *Neural Computing and Applications*, Springer.

11. Unsupervised real-time botnet detection. *arXiv:1804.05232*.

12. Blázquez-García, A. et al. (2020). A review on outlier/anomaly detection in time series data. *arXiv:2002.04236*.

13. Schmidl, S. et al. Anomaly Detection in Time Series: A Comprehensive Evaluation. *TimeEval* (timeeval.github.io/evaluation-paper/).
