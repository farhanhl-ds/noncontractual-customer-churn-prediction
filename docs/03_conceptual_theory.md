# Conceptual Theory

> **Disclaimer:** This document is an intentionally simplified explanation of the BG/NBD and Gamma-Gamma models,
> written to build intuition for readers without a marketing or statistics background.
> It omits mathematical formalism, likelihood derivations, and technical assumptions covered in the original papers.
> The goal is to give you enough intuition to understand *why* these models work and *what* their outputs mean
> before looking at any code. For the full theoretical treatment, refer directly to Fader et al. (2005)
> and Fader & Hardie (2013) linked in the References section.

---

## 1. The Core Problem: We Never Know When a Customer Leaves

Imagine you run an online store. A customer bought from you 3 months ago and hasn't
purchased since. Are they gone forever, or just taking a break?

You have no idea — and that's the fundamental challenge.

In some businesses this is easy: a gym member cancels their subscription, a phone
contract ends, a user deletes their account. These are **contractual** settings —
churn is an observable event.

But in a marketplace, e-commerce store, or any business where customers come and go
freely, there is no cancellation signal. A customer just... stops. This is a
**non-contractual** setting.

The question we're trying to answer is not:
> *"Has this customer churned?"* (we can't know for certain)

But rather:
> *"What's the probability this customer is still active?"*

That shift — from a yes/no answer to a probability — is the entire foundation of this project.

---

## 2. The Big Idea: Every Customer Lives and Dies

To model this, we make a simple but powerful assumption:

> Every customer goes through two phases in their relationship with a business:
> an **active phase** where they buy from time to time, and an **inactive phase**
> where they will never buy again. The tricky part is we never know exactly when
> they switched from active to inactive.

This is called the **Buy Till You Die (BTYD)** framework. "Die" here just means
"permanently stop buying" — not literally.

Think of it like a lightbulb. It works for a while, then at some random point it
burns out. You don't know exactly when it will burn out, but you can use patterns
(how often it flickers, how long it's been on) to estimate whether it's still
working or has already gone dark.

---

## 3. BG/NBD Model — How It Works

The BG/NBD model is the specific BTYD model we use. It tracks two things per customer simultaneously:

### 3.1 How Often They Buy (While Active)

While a customer is still active, they buy at their own personal pace — some buy every week,
some buy every few months. The model assumes this pace is **random but consistent** for each
individual, like rolling a dice at a fixed (but unknown) frequency.

Across all customers, these individual paces form a spectrum — some are naturally fast buyers,
some are slow. The model captures this spread so it doesn't assume everyone behaves the same way.

### 3.2 When They Drop Out

After each purchase, there's a small chance the customer quietly walks away forever.
This dropout probability is also individual — some customers are very loyal (low dropout chance),
others try a store once and never return (high dropout chance).

Again, the model accounts for the fact that different customers have different loyalty levels,
rather than assuming everyone is equally likely to leave.

### 3.3 The Clever Part: Both Happen at the Same Time

The real insight of BG/NBD is that it handles both processes **simultaneously from just
three numbers per customer:**

| Input | What it means |
|-------|--------------|
| `frequency` | How many times they came back after their first purchase |
| `recency` | How long ago their last purchase was (relative to their first) |
| `T` | How long they've been a customer overall |

From just these three numbers, the model estimates the probability that a customer
is still alive today — called **`p_alive`**.

### 3.4 The Silence Tells a Story

One of the most intuitive outputs of the model is how it handles silence differently
depending on who's silent:

| Customer profile | What silence means |
|---|---|
| Bought 20 times, last purchase 6 months ago | Very suspicious — this customer was active, now they've gone quiet. Probably churned. |
| Bought 2 times, last purchase 6 months ago | Less informative — this customer was always infrequent, so a 6-month gap is normal for them. |
| Bought 20 times, last purchase last week | No concern at all — still clearly active. |

In plain English: **silence from a chatty customer is a red flag. Silence from a quiet customer is normal.**

This asymmetry is something classical ML models struggle with because they don't have
the theoretical structure to reason about it — BG/NBD gets it right by design.

---

## 4. Gamma-Gamma Model — How Much Will They Spend?

Once we know *whether* a customer is likely to come back (BG/NBD), the next question is
*how much* they'll spend when they do.

The **Gamma-Gamma model** answers this. It's a simpler model that estimates each customer's
expected average spend per transaction, based on what they've spent historically.

The one important assumption is that **how often someone buys and how much they spend
are unrelated**. A frequent buyer isn't necessarily a big spender, and vice versa.
We verify this holds in the data before using the model.

### Putting It Together: CLV

Combining BG/NBD and Gamma-Gamma gives us **Customer Lifetime Value (CLV)** — an estimate
of how much revenue a customer is expected to generate over the next N weeks:

```
CLV = (expected number of future purchases) × (expected spend per purchase)
```

This is a forward-looking dollar value per customer, which is directly actionable for
decisions like how much to spend on a win-back campaign.

---

## 5. What the Model Parameters Actually Mean

When BG/NBD is fitted, it returns four numbers. Here's what they mean in plain English:

| Parameter | Plain English |
|-----------|--------------|
| `r` | How spread out purchase rates are across customers — lower means bigger spread between fast and slow buyers |
| `alpha` | The overall baseline purchase rate — higher means customers buy less frequently on average |
| `a` | How spread out dropout probabilities are across customers |
| `b` | The baseline loyalty level — higher `b` relative to `a` means customers tend to stick around longer |

The ratio `a / (a + b)` gives the average dropout probability across all customers.
If `a < b`, most customers are loyal. If `a > b`, most customers are transient.

---

## 6. Where the Probabilistic Model Falls Short

BG/NBD is elegant but it only knows three things about each customer: frequency, recency,
and tenure. It has no idea about:

- What products they bought
- Whether they're a new customer or a long-term one
- Whether they're from a high-value segment
- Whether they've been browsing but not buying lately

This is a real limitation. Two customers can have identical RFM profiles but very different
churn risk if one of them has been spending less per visit over time, or buying from
lower-margin categories.

---

## 7. Why We Add a Machine Learning Layer

This is exactly why we don't stop at BG/NBD. The **hybrid approach** combines the best
of both worlds:

**What BG/NBD contributes:**
It solves the hardest part of the problem — generating a theoretically sound churn label
(`p_alive`) without needing any explicit "this customer churned" ground truth. Without this,
we'd have no labels to train an ML model on at all.

**What ML contributes:**
Once we have labels, a classifier like XGBoost can take in dozens of behavioral features —
product diversity, spend trends, purchase regularity, basket size — and find patterns that
pure RFM can't capture.

**The combination:**
```
BG/NBD  →  p_alive score  →  churn label (1 if p_alive < threshold, else 0)
                                    ↓
                        XGBoost / LightGBM
                        + full behavioral features
                                    ↓
                        Churn risk score per customer
```

**The one risk to be aware of:**
If BG/NBD fits the data poorly, the labels it generates will be noisy, and the ML model
will learn from incorrect supervision. This is why we spend significant effort validating
the BG/NBD fit before ever training the classifier.

---

## References

- Fader, P.S., Hardie, B.G.S., & Lee, K.L. (2005).
  ["Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model.](http://www.brucehardie.com/papers/018/fader_et_al_mksc_05.pdf)
  *Marketing Science*, 24(2), 275–284.

- Fader, P.S. & Hardie, B.G.S. (2013).
  [The Gamma-Gamma Model of Monetary Value.](http://www.brucehardie.com/notes/025/gamma_gamma.pdf)

- Hardie, B.G.S. (2014).
  [Notes on the BG/NBD Model.](http://www.brucehardie.com/notes/004/bgnbd_spreadsheet_note.pdf)

- [`lifetimes` library documentation](https://lifetimes.readthedocs.io)

- [`pymc-marketing` CLV documentation](https://www.pymc-marketing.io/en/stable/notebooks/clv/bg_nbd.html)
