# Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretability

The Basel II Accord requires financial institutions to measure, document, and justify the credit risk associated with lending decisions. Under the Internal Ratings-Based (IRB) approach, banks must produce:

* Clear risk estimates
* Transparent, auditable model logic
* Consistent documentation of model assumptions
* Ability to explain decisions to regulators and auditors

Because of this, a credit scoring model cannot be a “black box.”
Even if advanced ML models perform better, the bank must always be able to explain why the model assigns a customer as high-risk or low-risk.

Therefore, the model we build must:

✔️ Be interpretable
✔️ Produce stable risk estimates
✔️ Include full documentation of assumptions, features, and decisions
✔️ Comply with regulatory requirements on fairness and transparency

---

### 2. Why a Proxy Target Variable Is Required

The dataset does not include a direct default label (e.g., “customer did not repay loan”).
Since we cannot build a supervised model without a target, we must:

* Engineer a proxy variable based on behavioral patterns
* Use Recency, Frequency, and Monetary (RFM) metrics to identify disengaged customers
* Define the least active RFM cluster as high-risk (1) and others as low-risk (0)

This is common in alternative credit scoring systems where loan histories do not exist.

#### Business Risks of Using a Proxy Target

* Mislabeling customers: A low-spender may be incorrectly classified as “high risk.”
* Bias risk: Clusters may reflect income or demographic patterns, not true risk.
* Regulatory concerns: Proxy-based models must be validated before production.
* Model drift: Customer engagement patterns may change over time.

Banks must re-evaluate this proxy once real repayment data becomes available.

---

### 3. Trade-Offs: Simple vs. Complex Models
3. Trade-Offs: Simple vs. Complex Models

When building predictive models for insurance risk, you often face a choice between using simple models or more complex ones. Each comes with benefits and drawbacks, and neither is universally “better”—the best option depends on the business need.

Simple models, such as linear regression or logistic regression, are easy to interpret and quick to train. They can reveal clear relationships between variables, making them ideal when leadership needs transparency or when the goal is to explain why certain customers are high-risk. However, their simplicity also limits them—they may miss important nonlinear patterns or interactions within the data.

Complex models, such as Random Forest, Gradient Boosting, or Neural Networks, capture deeper patterns and usually provide higher predictive accuracy. These models can handle messy interactions, large feature spaces, and nonlinear relationships far better than simple models. But they sacrifice interpretability: it is harder to explain individual predictions to business teams, regulators, or customers. They also require more computational resources, parameters, and careful tuning.

Choosing between the two depends on the priorities of the project. If interpretability, speed, or regulatory transparency matter most, simpler models are the right choice. But when the main objective is maximizing prediction quality—such as detecting high-risk insurance customers more accurately—complex models often outperform simpler alternatives. In practice, businesses often use both: simple models for interpretability and stakeholder communication, and complex models for operational decision-making where accuracy is critical.

In a financial-service context, the bank must balance:

* Accuracy (business performance)
* Interpretability (regulatory compliance)
* Operational stability (easy monitoring and updating)

Because of these constraints, Logistic Regression with WoE/IV is often preferred for production, while complex models may be used only for challenger or experimental scoring.
