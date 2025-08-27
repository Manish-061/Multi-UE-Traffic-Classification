# 📚 Multi-UE Traffic Classifier: Data Guide

Welcome to the Data Guide for the Multi-UE Traffic Classifier project. Understanding the data is key to understanding the model. This document details the data sources, features, and the strategic splitting used to build a robust classifier.

---

## Data Sources

The model is trained on a hybrid of public and synthetically generated datasets to ensure diversity and balance.

1.  **Public Datasets**: We use well-known academic datasets for a baseline of realistic network traffic. These are specified in `configs/data.yaml` and include:
    -   **CIC-IDS2017**: A popular dataset for intrusion detection, containing a wide variety of modern network traffic.
    -   **UNSW-NB15**: Another comprehensive dataset with a good mix of normal and malicious traffic patterns.

2.  **Synthetic Data**: To ensure the model learns the specific signatures of our target application classes, we built a powerful synthetic data generator (`scripts/generate_synthetic_data.py`). This allows us to create a large, perfectly labeled dataset where we can control the precise characteristics of each traffic type.

3.  **Live Traffic Capture**: The system is also capable of processing live network traffic from a specified interface, making it adaptable for real-world deployment.

---

## 📊 Flow Features

The core of our classification approach relies on statistical features extracted from network flows. A "flow" is a sequence of packets between two endpoints. We use `CICFlowMeter` to generate over 70 statistical features. The most impactful ones, configured in `configs/features.yaml`, are grouped below.

### Key Feature Groups

| Feature Group | Description | Example Features | Keyword | 
| :--- | :--- | :--- | :--- |
| **Flow Statistics** | Basic counts and sizes of packets and bytes. | `Total Fwd Packets`, `Total Length of Bwd Packets` | `Volume` |
| **Timing Features** | Captures the rhythm and timing of packet exchange. | `Flow IAT Mean`, `Fwd IAT Std`, `Flow Duration` | `Pacing` |
| **Behavioral Features** | Describes the high-level behavior of the flow. | `Fwd PSH Flags`, `Down/Up Ratio`, `Avg Packet Size` | `Behavior` |
| **Advanced Flags** | Counts of specific TCP flags, indicating connection state changes. | `FIN Flag Count`, `SYN Flag Count`, `RST Flag Count` | `Control` |


### Synthetic Data Characteristics

Our synthetic generator models each application class with distinct, tunable characteristics. This is crucial for teaching the model the subtle differences between them.

-   **🎮 Gaming**:
    -   **Keyword**: `Bursty & Low-Latency`
    -   **Description**: Characterized by small, frequent packets with very low inter-arrival times. The model learns to associate this bursty, low-volume pattern with gaming.

-   **📺 Video Streaming**:
    -   **Keyword**: `High & Sustained Throughput`
    -   **Description**: Involves large data packets sent from server to client at a high, consistent rate. The model identifies this high-volume, long-duration signature.

-   **📞 Audio & Video Calls**:
    -   **Keyword**: `Consistent & Bidirectional`
    -   **Description**: Traffic is typically bidirectional and consistent over time. Video calls have a significantly higher data rate than audio-only calls.

-   **🌐 Browsing**:
    -   **Keyword**: `Request-Response Pattern`
    -   **Description**: Short bursts of client-to-server requests followed by larger server-to-client responses. Flow durations are often short.

---

## 🔪 Data Splitting Strategy: UE-Based Splitting

To simulate a real-world scenario and prevent data leakage, we do **not** use a random split.

**The Problem with Random Splitting**: In a network, all traffic from a single user (UE) is correlated. If you randomly sprinkle a user's flows into the training, validation, *and* test sets, the model can "cheat" by learning to identify the user instead of the traffic type. This leads to an artificially inflated performance score that doesn't generalize.

**Our Solution**: We use a **UE-Based Splitting** strategy, implemented in `src/traffic_classifier/data/splitter.py`.

1.  **Group by UE**: All flows are first grouped by their `UE_ID`.
2.  **Split UEs, Not Flows**: The list of unique UEs is split into three groups: one for training, one for validation, and one for testing.
3.  **Create Datasets**: The final datasets are constructed from the flows belonging to the UEs in each group.

This ensures that the model is tested on data from users it has **never seen before**, providing a much more accurate and reliable measure of its real-world performance.

### Final Datasets

This process results in the following key files, located in `data/splits/`:

-   `train_data.csv`: For training the model.
-   `val_data.csv`: For hyperparameter tuning and calibration.
-   `test_data.csv`: For the final, unbiased evaluation of the model.
-   `ue_assignments.json`: A record of which UEs were assigned to each dataset.
