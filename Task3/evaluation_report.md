# Homework 3 — Task 3: Evaluation & Diagnostic Report

## 1. Primary & Secondary Metric Interpretation
- **Primary Metric (Macro-Averaged F1 = 0.4349)**: Reflects unweighted mean performance across all routing departments. A baseline Macro F1 of $0.4349$ indicates that while the model captures distinctive lexical cues for major queues, minor or semantically overlapping categories require further feature tuning to avoid performance starvation.
- **Secondary Metric (Categorical Log Loss = 1.5898)**: Evaluates probability score calibration. The observed log loss verifies that predicted class probability distributions align reasonably with true labels, serving as a reliable filter for automated threshold routing.

---

## 2. Departmental Performance & Confusion Dynamics
- **High-Performing Queues**: Departments with distinct keyword spaces (e.g., *Billing*) exhibit superior precision and recall.
- **Inter-Departmental Confusion**: Misclassifications predominantly concentrate between adjacent operational queues (e.g., *Technical Support* vs. *Bug Reports*), where user phrasing shares high semantic overlap.
- **Prior Imbalance Effects**: Class-prior imbalances cause high-frequency general queues to absorb ambiguous edge cases, inflating false negatives in specialized technical categories.

---

## 3. Slice Performance Analysis (Body Length)
- **Slice Breakdown Findings**:
  - **Short Tickets ($N=1246$)**: Macro F1 = $0.3901$ | Log Loss = $1.6948$. Exhibit depressed performance due to feature sparsity; brief text limits the TF-IDF vectorizer's capacity to extract robust discriminative tokens.
  - **Medium Tickets ($N=1232$)**: Macro F1 = $0.4782$ | Log Loss = $1.5172$. Achieves peak performance, containing optimal contextual depth.
  - **Long Tickets ($N=1277$)**: Macro F1 = $0.4428$ | Log Loss = $1.5576$. Demonstrate a slight degradation relative to medium tickets, driven by conversational noise, boilerplate disclaimers, and multi-issue dilution.

---

## 4. Limitations: What This Validation Set Cannot Establish
1. **Out-of-Distribution (OOD) Robustness**: Static offline splits cannot account for production drift caused by sudden product updates, system outages, or changing customer behavior.
2. **Downstream Economic Impact**: Offline metrics measure classification accuracy but fail to capture operational trade-offs such as agent re-triage overhead, customer resolution latency, and SLA compliance.
3. **Temporal Drift**: Validation assumes stationarity and does not model concept drift over extended deployment horizons.