## Loss Functions for Count Data

When modeling count data, choosing the right loss function is crucial.

### 1. **Mean Squared Error (MSE)**

Measures the squared difference between true (\(y\)) and predicted (\(\hat{y}\)) values:

\[
\text{MSE}(y,\hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

- **Assumption**: Errors are normally distributed (constant variance).
- **Pros:** Simple, intuitive.
- **Cons:** Not ideal for discrete count data; assumes constant variance.

---

### 2. **Huber Loss**

A robust version of MSE that penalizes large errors linearly instead of quadratically:

\[
L_{\delta}(y, \hat{y}) = 
\begin{cases} 
\frac{1}{2}(y-\hat{y})^2 & \text{if } |y-\hat{y}| \leq \delta \\[6pt]
\delta \cdot (|y-\hat{y}| - \frac{1}{2}\delta) & \text{otherwise}
\end{cases}
\]

- \(\delta\) controls transition between quadratic and linear behavior.

---

### **Probabilistic Losses for Count Data**

### **Negative Poisson Log-Likelihood (NPLL)**

Assumes each data point is Poisson-distributed. For count \(y\) and prediction \(\lambda\):

\[
\text{NPLL}(y, \lambda) = \lambda - y\log(\lambda + \epsilon)
\]

- Assumes **variance = mean** (Poisson assumption).

---

### **Negative Binomial (NB) Loss**

Generalizes Poisson, allowing **overdispersion** (variance exceeds the mean):

For count \(y\), predicted mean \(\mu\), and dispersion \(\theta\):

\[
\text{NB-NLL}(y,\mu,\theta) = -\log\left[
\frac{\Gamma(y+\theta)}{\Gamma(\theta)\Gamma(y+1)}\left(\frac{\theta}{\mu+\theta}\right)^\theta\left(\frac{\mu}{\mu+\theta}\right)^y
\right]
\]

- Explicitly models **overdispersion**: variance = \(\mu + \frac{\mu^2}{\theta}\).

---

### ⚠️ **Overdispersion (Important Concept)**

- **Poisson** assumes variance = mean.
- **Overdispersion**: Observed variance > mean, common in genomic or biological counts.
- When overdispersion is detected, **Negative Binomial loss** is preferred over Poisson.

**How to detect overdispersion?**

Check if variance significantly exceeds mean in residuals:

- Calculate standardized residuals:
\[
r_{ij}^{std} = \frac{y_{ij}-\hat{y}_{ij}}{\sqrt{\hat{y}_{ij}}}
\]

If variance of these residuals significantly exceeds 1, overdispersion is present.

---

This table summarizes clearly:

| Loss                     | Assumption                      | Handles Overdispersion?  |
|--------------------------|---------------------------------|--------------------------|
| MSE / Huber              | Continuous errors, normality    | ❌ No                    |
| Poisson NLL              | Variance = Mean                 | ❌ No                    |
| Negative Binomial NLL    | Variance ≥ Mean                 | ✅ Yes                   |

---
