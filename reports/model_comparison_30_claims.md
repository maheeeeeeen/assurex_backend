# AssureX Claim Engine — Model Comparison Benchmark (Deliverable 6)

Evaluation of 35 unseen test claims across the dual AI architecture:
1. **Python Tabular Classifier (`XGBoost`)**
2. **Google Teachable Machine Image Classifier (`MobileNetV2`)**

## 1. Executive Summary

- **Evaluated Claims:** 35 unseen test claims
- **Overall Agreement Rate:** **94.29%** (33/35 identical class decisions)
- **Mean Absolute Confidence Difference:** **12.61%** (`|Tabular Conf - TM Conf|`)
- **Python Tabular Accuracy:** **100.0%**
- **Teachable Machine Accuracy:** **94.29%**

### Match Category Breakdown

| Category | Criteria | Count | Percentage |
|---|---|---|---|
| **Strong Match** | Same class, $\Delta$ < 5% | 15 | 42.9% |
| **Acceptable Match** | Same class, 5% $\le \Delta \le$ 15% | 4 | 11.4% |
| **Weak Match** | Same class, 15% < $\Delta \le$ 25% | 11 | 31.4% |
| **Weak Match (>25% gap)** | Same class, $\Delta$ > 25% | 3 | 8.6% |
| **Disagreement** | Differing predicted classes | 2 | 5.7% |
| **Uncertain** | Top confidence < 50% | 0 | 0.0% |

## 2. Granular 35-Claim Comparison Table

| # | Claim ID | Category | Ground Truth | Tabular Pred (Conf) | TM Pred (Conf) | Diff ($\Delta$) | Agreement | Match Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `CLM-01995` | Appliances | **Likely Valid** | Likely Valid (99.7%) | Likely Valid (80.6%) | 19.0% | ✅ Yes | Weak Match |
| 2 | `CLM-00775` | Appliances | **Manual Review Required** | Manual Review Required (99.7%) | Likely Valid (69.0%) | 30.7% | ❌ No | Disagreement |
| 3 | `CLM-00062` | Appliances | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (96.7%) | 3.0% | ✅ Yes | Strong Match |
| 4 | `CLM-00107` | Electronics | **Likely Invalid** | Likely Invalid (99.7%) | Likely Invalid (95.9%) | 3.8% | ✅ Yes | Strong Match |
| 5 | `CLM-00456` | Appliances | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (84.3%) | 15.5% | ✅ Yes | Weak Match |
| 6 | `CLM-02076` | Appliances | **Likely Valid** | Likely Valid (99.8%) | Likely Valid (91.1%) | 8.7% | ✅ Yes | Acceptable Match |
| 7 | `CLM-02354` | Appliances | **Likely Valid** | Likely Valid (99.8%) | Likely Valid (70.1%) | 29.7% | ✅ Yes | Weak Match (>25% gap) |
| 8 | `CLM-01916` | Electronics | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (75.2%) | 24.7% | ✅ Yes | Weak Match |
| 9 | `CLM-00178` | Electronics | **Manual Review Required** | Manual Review Required (99.5%) | Manual Review Required (97.8%) | 1.6% | ✅ Yes | Strong Match |
| 10 | `CLM-00276` | Electronics | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (82.3%) | 17.5% | ✅ Yes | Weak Match |
| 11 | `CLM-00713` | Automotive | **Likely Valid** | Likely Valid (99.8%) | Likely Valid (76.1%) | 23.8% | ✅ Yes | Weak Match |
| 12 | `CLM-02268` | Automotive | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (83.8%) | 16.1% | ✅ Yes | Weak Match |
| 13 | `CLM-01559` | Electronics | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (92.1%) | 7.6% | ✅ Yes | Acceptable Match |
| 14 | `CLM-00535` | Appliances | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (98.5%) | 1.3% | ✅ Yes | Strong Match |
| 15 | `CLM-02103` | Electronics | **Manual Review Required** | Manual Review Required (99.0%) | Likely Valid (75.9%) | 23.1% | ❌ No | Disagreement |
| 16 | `CLM-01987` | Electronics | **Likely Valid** | Likely Valid (99.8%) | Likely Valid (83.5%) | 16.3% | ✅ Yes | Weak Match |
| 17 | `CLM-01876` | Appliances | **Likely Invalid** | Likely Invalid (99.7%) | Likely Invalid (99.7%) | 0.0% | ✅ Yes | Strong Match |
| 18 | `CLM-00785` | Electronics | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (75.0%) | 24.9% | ✅ Yes | Weak Match |
| 19 | `CLM-00283` | Appliances | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (80.1%) | 19.7% | ✅ Yes | Weak Match |
| 20 | `CLM-02139` | Electronics | **Manual Review Required** | Manual Review Required (99.5%) | Manual Review Required (60.5%) | 39.0% | ✅ Yes | Weak Match (>25% gap) |
| 21 | `CLM-00543` | Electronics | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (80.1%) | 19.8% | ✅ Yes | Weak Match |
| 22 | `CLM-00684` | Electronics | **Likely Valid** | Likely Valid (99.9%) | Likely Valid (75.6%) | 24.2% | ✅ Yes | Weak Match |
| 23 | `CLM-01150` | Electronics | **Manual Review Required** | Manual Review Required (99.2%) | Manual Review Required (90.5%) | 8.7% | ✅ Yes | Acceptable Match |
| 24 | `CLM-01657` | Electronics | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (96.5%) | 3.4% | ✅ Yes | Strong Match |
| 25 | `CLM-00973` | Automotive | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (98.9%) | 0.9% | ✅ Yes | Strong Match |
| 26 | `CLM-00393` | Electronics | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (94.5%) | 5.2% | ✅ Yes | Acceptable Match |
| 27 | `CLM-02465` | Automotive | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (99.1%) | 0.7% | ✅ Yes | Strong Match |
| 28 | `CLM-01216` | Automotive | **Manual Review Required** | Manual Review Required (99.5%) | Manual Review Required (96.8%) | 2.7% | ✅ Yes | Strong Match |
| 29 | `CLM-02348` | Automotive | **Manual Review Required** | Manual Review Required (99.5%) | Manual Review Required (61.3%) | 38.2% | ✅ Yes | Weak Match (>25% gap) |
| 30 | `CLM-01745` | Automotive | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (99.9%) | 0.1% | ✅ Yes | Strong Match |
| 31 | `CLM-01068` | Appliances | **Likely Invalid** | Likely Invalid (99.8%) | Likely Invalid (99.1%) | 0.7% | ✅ Yes | Strong Match |
| 32 | `CLM-01974` | Electronics | **Manual Review Required** | Manual Review Required (99.4%) | Manual Review Required (96.8%) | 2.6% | ✅ Yes | Strong Match |
| 33 | `CLM-00856` | Automotive | **Manual Review Required** | Manual Review Required (99.4%) | Manual Review Required (96.8%) | 2.6% | ✅ Yes | Strong Match |
| 34 | `CLM-02283` | Automotive | **Manual Review Required** | Manual Review Required (99.4%) | Manual Review Required (95.8%) | 3.6% | ✅ Yes | Strong Match |
| 35 | `CLM-00578` | Appliances | **Manual Review Required** | Manual Review Required (99.6%) | Manual Review Required (97.6%) | 2.0% | ✅ Yes | Strong Match |

## 3. Key Observations & Synergy Analysis

1. **Complementary Decision Validation:** When both the structured tabular engine and the computer vision card classifier agree with high confidence, the system achieves near 100% precision, qualifying the claim for zero-touch **Auto-Approval** or **Auto-Rejection**.
2. **Disagreement Detection:** Claims exhibiting divergence between models automatically trigger escalation to the **Manual Review Queue**, preventing false positives and fraudulent payouts.
3. **Retina Card High-DPI Legibility:** Because Claim Summary Cards are rendered at 1200x1680 vector resolution with zero text overlap, MobileNetV2 extracts pristine spatial layout features matching human visual adjudication.
