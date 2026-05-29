# Problem Framing

## 1. The Business Context

Customer churn is one of the most expensive problems in any business that depends on
repeat purchases. Acquiring a new customer typically costs 5–7x more than retaining
an existing one. Identifying customers who are about to leave — before they actually
leave — gives a business the window to act: offer a discount, send a re-engagement
campaign, or prioritize customer service resources.

But not all churn looks the same. How churn *manifests* depends entirely on the type
of business, and that difference has major implications for how we model it.

---

## 2. Contractual vs Non-Contractual Settings

| | Contractual | Non-Contractual |
|---|---|---|
| **Examples** | Subscriptions, SaaS, telecom, insurance | Marketplaces, e-commerce, retail |
| **Churn signal** | Explicit — customer cancels | None — customer just stops buying |
| **Churn label** | Directly observable | Latent — must be inferred |
| **Modeling approach** | Standard binary classifier | Probabilistic + behavioral inference |

In a **contractual** setting, churn is an event you can observe and timestamp.
A Netflix subscriber cancels. A SaaS user downgrades to free. A phone contract expires.
You have a clean label: churned = 1, not churned = 0. Any classification model works.

In a **non-contractual** setting — a marketplace, an online retailer, a wholesale supplier —
no such event exists. A customer who bought 3 months ago and hasn't returned could be:
- Genuinely churned and shopping elsewhere
- Waiting for the right product to come back in stock
- On a natural long inter-purchase cycle
- About to place a large order next week

There is no way to know for certain. This is the defining challenge of non-contractual churn.

---

## 3. Why This Is Harder Than It Looks

The instinct is to define a rule: *"if a customer hasn't bought in 90 days, they've churned."*
This is called a **fixed inactivity threshold**, and it's the most common naive approach.

The problem is it treats all customers the same. A customer who normally buys every
2 weeks going silent for 90 days is very different from a customer who buys twice a year
going silent for 90 days. The same threshold produces very different false positive
and false negative rates across different customer segments.

A more principled approach is to ask: *given this customer's entire purchase history,
what is the probability they are still active today?* That framing leads naturally to
a probabilistic model — one that accounts for individual behavioral patterns rather
than applying a one-size-fits-all cutoff.

---

## 4. Why Classical ML Models Fall Short

Standard binary classifiers (logistic regression, random forest, XGBoost) require
a labeled training set: each customer must be tagged as churned or not churned.

In non-contractual settings this label simply does not exist. You cannot label a customer
as churned without first deciding they've churned — which is exactly what you're trying
to predict. This creates a **circular dependency**:

```
Need label to train model → Need model to generate label
```

Any workaround — fixed thresholds, arbitrary time windows — introduces bias that
the model then learns and perpetuates. The predictions become a reflection of your
labeling rule, not the actual customer behavior.

---

## 5. The Right Framing: From Labels to Probabilities

The solution is to abandon the idea of a hard churn label entirely and instead model
the underlying process that generates purchase behavior.

Rather than asking *"has this customer churned?"*, we ask:
> *"Given everything we know about this customer's purchase history, what is the
> probability they are still an active customer today?"*

This reframing has two important consequences:

1. **It produces a continuous risk score** (0–1) rather than a binary label,
   which is more useful for business decisions — you can prioritize the top 10%
   most at-risk customers rather than working from an arbitrary churn/not-churn split.

2. **It makes the modeling tractable** — there are well-established probabilistic
   models designed specifically for this problem (BG/NBD, Pareto/NBD) that have
   been validated empirically across many industries.

---

## 6. Business Impact

Getting non-contractual churn right has direct revenue implications:

| Use case | How churn scores help |
|---|---|
| **Win-back campaigns** | Target high-CLV customers with low `p_alive` before they're fully lost |
| **Retention budget allocation** | Focus spend on "At Risk" segment, not customers who are already gone |
| **Inventory & demand planning** | Expected future purchases per customer informs demand forecasts |
| **Customer health dashboards** | Real-time monitoring of customer base vitality |
| **Sales prioritization** | Sales teams focus outreach on customers with declining `p_alive` |

---

## 7. Our Approach

This project addresses non-contractual churn in two stages, designed to complement each other:

**Stage 1 — Probabilistic baseline (BG/NBD + Gamma-Gamma)**
Uses only transaction history (frequency, recency, customer tenure) to estimate
the probability each customer is still alive and their expected lifetime value.
Theoretically grounded, interpretable, and requires minimal data.

**Stage 2 — Hybrid ML extension (XGBoost / LightGBM)**
Uses BG/NBD-derived `p_alive` scores as pseudo churn labels to train a classifier
that incorporates richer behavioral features. Captures patterns that pure RFM cannot —
product diversity, spend trends, purchase regularity, and more.

The two stages are complementary: the probabilistic layer provides the theoretical
foundation and solves the label problem; the ML layer provides predictive power and
feature richness. See [`03_conceptual_theory.md`](03_conceptual_theory.md) for the
theory behind the models and [`04_methodology.md`](04_methodology.md) for how they
are applied in this project.
