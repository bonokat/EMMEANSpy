# EMMEANSpy

**Estimated Marginal Means (EMMs) for Python**

EMMEANSpy is a Python implementation of estimated marginal means (EMMs), also known as least-squares means, inspired by the R **emmeans** package. It is designed to work with `statsmodels` models while providing a familiar workflow for users transitioning from R.

The package computes estimated marginal means, pairwise comparisons, and custom contrasts for linear and generalized linear models, with statistical behaviour closely matching that of R's **emmeans**.

## Features

* Compute estimated marginal means (EMMs)
* Pairwise and custom contrasts
* Support for continuous covariates via reference grids
* Equal or proportional marginal weighting
* Multiple comparison corrections (Bonferroni, Holm, Sidak, Tukey*)
* Confidence intervals
* Support for both linear models (LM) and generalized linear models (GLMs)
* Automatic handling of link and response scales
* Output designed to closely resemble R **emmeans**

* Tukey adjustment is available where the required assumptions are met.

---

## Installation

```bash
pip install emmeanspy
```

Or install from source:

```bash
git clone https://github.com/bonokat/EMMEANSpy.git
cd EMMEANSpy
pip install -e .
```

---

## Quick example

```python
import statsmodels.formula.api as smf
import emmeans

model = smf.glm(
    "y ~ treatment * load + age",
    data=df,
    family=sm.families.Gamma(link=sm.families.links.Log())
).fit()

results = emmeans.emmeans(
    model,
    data=df,
    specs="treatment",
    transform="response",
    contrasts="pairwise",
    adjust="bonferroni"
)

print(results["emm"])
print(results["contrasts"])
```

---

## Documentation

The repository includes several tutorial notebooks covering:

* Basic estimated marginal means
* Generalized linear models
* Pairwise contrasts
* Interaction contrasts
* Multiple comparison corrections
* Reference grids
* Custom contrasts
* Working with continuous covariates

---

## Relationship to R emmeans

EMMEANSpy is heavily inspired by the excellent R package **emmeans** developed by Russell Lenth.

The goal is not to be a direct port of the R code, but to reproduce the underlying statistical methodology and provide a familiar interface for Python users.

Where possible, output has been designed to closely match R **emmeans**, including:

* estimated marginal means
* confidence intervals
* link-scale hypothesis tests
* response-scale summaries
* pairwise comparisons
* multiple-comparison adjustments

---

## Requirements

* Python ≥ 3.10
* NumPy
* pandas
* SciPy
* patsy
* statsmodels

---

## Citation

If you use EMMEANSpy in academic work, please cite the repository and the original R **emmeans** package.

Lenth RV. *emmeans: Estimated Marginal Means, aka Least-Squares Means*. R package.

---

## License

MIT License.
