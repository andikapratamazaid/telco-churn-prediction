# Customer Churn Prediction for a Telecom Provider

> Predicting which telecom customers are about to cancel, and choosing *who to contact* so that a retention campaign is profitable. Built with scikit-learn.

**Dataset:** IBM Telco Customer Churn (7,043 customers, 21 features) · **Tools:** Python, pandas, scikit-learn, matplotlib, seaborn

## Business Problem
A telecom provider loses revenue every time a customer cancels, and retention offers cost money. The question is not only "who will churn?" but **"which customers should we contact so the campaign earns more than it costs?"**

## Approach
1. **Data validation:** fix `TotalCharges` stored as text, confirm the blank rows are brand-new customers (`tenure = 0`), check duplicates.
2. **EDA:** churn rate by contract, internet service, payment method and tenure.
3. **Feature engineering:** number of add-on services, autopay flag, average charge per tenure month.
4. **Modeling:** logistic regression baseline vs. tuned gradient boosting, 5-fold stratified cross-validation, metrics focused on PR-AUC because the classes are imbalanced.
5. **Business-driven threshold:** the decision threshold maximizes expected campaign profit (retention cost, offer success rate, margin), chosen on out-of-fold predictions, not on the test set.
6. **One-shot test evaluation:** ROC/PR curves, confusion matrix, lift table.
7. **Explainability:** permutation importance and partial dependence plots.

## Key Decisions
| Issue | Decision |
|---|---|
| Churn is imbalanced (roughly a quarter of customers) | Use PR-AUC and a profit-based threshold; accuracy is not reported as the main metric |
| Blank `TotalCharges` values | Verified they are `tenure = 0` customers, then set to 0 |
| Risk of data leakage | Test set split first; all preprocessing inside a `Pipeline` |
| Complex model vs. simple model | Gradient boosting is kept only if it clearly beats logistic regression |
| Threshold at 0.5 is arbitrary | Threshold chosen by expected profit, with stated assumptions |

## Project Structure
```
notebooks/churn_prediction.ipynb   -- full analysis, from data validation to model saving
data/                              -- place the CSV here (not committed)
models/                            -- saved model + threshold (generated)
images/                            -- charts for this README
requirements.txt
```

## How to Run
```bash
git clone https://github.com/<username>/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt
# download the dataset into data/ (see data/README.md)
jupyter notebook notebooks/churn_prediction.ipynb
```

## Results  *(fill in after running the notebook; use your real numbers)*
| Model | CV PR-AUC | Test ROC-AUC | Test PR-AUC |
|---|---|---|---|
| Logistic regression | ___ | ___ | ___ |
| Gradient boosting (tuned) | ___ | ___ | ___ |

- **Chosen threshold:** ___ → contacts ___% of customers, captures ___% of churners
- **Top 10% of scored customers** contain ___% of all churners (lift table)
- **Main churn drivers:** ___, ___, ___

![Model evaluation](images/evaluation.png)

## Business Recommendations *(write these from your own findings; example format)*
| Finding | Action |
|---|---|
| Customers on the riskiest contract type churn far more than average | Offer an incentive to move to a longer contract |
| New customers are the most likely to leave in the first months | Onboarding and early-life check-ins |
| A specific payment method is associated with higher churn | Encourage autopay enrollment |

## Limitations
- Single snapshot with no timestamps, so there is no temporal validation.
- Cost, success-rate and margin figures are **assumptions**; replace them with real company numbers.
- Predicting who churns is not the same as predicting who responds to an offer; an uplift model or an A/B test would address this.
- Probabilities are not explicitly calibrated.

## Next Steps
- SHAP explanations and a fairness check across customer groups
- Probability calibration
- Uplift modeling
- Monitoring plan for data drift

## Related Project
[E-Commerce RFM Segmentation & Cohort Retention (SQL)](https://github.com/<username>/ecommerce-rfm-cohort-sql)

## Author
**<YOUR NAME>** · [LinkedIn](https://linkedin.com/in/<username>) · [GitHub](https://github.com/<username>)
