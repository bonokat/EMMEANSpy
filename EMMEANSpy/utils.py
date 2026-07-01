"""
Estimated Marginal Means (EMMs) Implementation

This section implements a comprehensive `emmeans` function for post-hoc testing of mixed linear models.

"""

# Import EMM dependencies
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, List, Optional, Dict, Tuple

import numpy as np
import pandas as pd
import patsy
from itertools import product, combinations
from scipy.stats import norm, t as tdist
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import re

try:
    # statsmodels >= 0.14
    from statsmodels.stats.libqsturng import psturng
    _HAVE_PSTURNG = True
except Exception:
    _HAVE_PSTURNG = False

from pandas.api.types import CategoricalDtype

# ----------------------------- Utilities -----------------------------

def _groupby(df: pd.DataFrame, by_cols: List[str], **kwargs):
    """
    Wrapper that (a) avoids passing duplicate 'dropna' and
    (b) gracefully supports pandas < 1.1 where 'dropna' is unsupported.
    """
    dropna_kw = kwargs.pop("dropna", None)
    try:
        if dropna_kw is not None:
            return df.groupby(by_cols, dropna=dropna_kw, **kwargs)
        else:
            # default to dropna=False for stable marginalization behavior
            return df.groupby(by_cols, dropna=False, **kwargs)
    except TypeError:
        # pandas < 1.1: retry without 'dropna'
        return df.groupby(by_cols, **kwargs)


def _base_name(patsy_name: str) -> str:
    n = patsy_name.strip()
    if n.startswith("C(") and n.endswith(")"):
        return n[2:-1].strip()
    return n


def _extract_dep_vars(expr: str, data_cols: pd.Index) -> List[str]:
    """
    Heuristic: pull candidate symbols from an EvalFactor's name,
    keep only those that are actual columns in `data`.
    """
    tokens = set(re.findall(r"[A-Za-z_]\w*", expr))
    ignore = {"C", "I", "np", "pd", "math", "sin", "cos", "tan", "exp", "log", "sqrt"}
    return [t for t in tokens if (t not in ignore and t in data_cols)]


def _get_design_info_or_raise(result):
    try:
        return result.model.data.design_info
    except AttributeError:
        raise ValueError("Model must be fitted with a Patsy formula to have design_info.")


def _get_model_params_and_vcov(result, vcov: Optional[np.ndarray]) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Handles MixedLM vs others and aligns covariance to fixed-effect parameter order.
    """
    if hasattr(result, "fe_params"):  # MixedLM
        beta = result.fe_params.values
        param_names = list(result.fe_params.index)
        if vcov is not None:
            V = np.asarray(vcov)
        else:
            V_full = result.cov_params()
            if hasattr(V_full, "loc"):  # DataFrame
                V = V_full.loc[param_names, param_names].to_numpy()
            else:
                # ndarray fallback: assume order matches params; slice by position
                all_names = list(getattr(result, "params").index)
                idx = [all_names.index(n) for n in param_names]
                V = np.asarray(V_full)[np.ix_(idx, idx)]
    else:
        beta = result.params.values
        param_names = list(result.params.index)
        V = np.asarray(vcov) if vcov is not None else np.asarray(result.cov_params())
    return beta, param_names, V


def _get_link_and_deriv(result):
    """
    Returns
    -------
    link_name : str
        Name of the model link function (e.g. "log", "identity", "logit").
    invlink : callable
        Inverse link.
    deriv : callable
        Derivative of inverse link wrt eta.
    """
    try:
        link = result.model.family.link
        link_name = type(link).__name__.lower()

        invlink = link.inverse

        def deriv(eta):
            if hasattr(link, "inverse_deriv"):
                return link.inverse_deriv(eta)

            eps = 1e-6
            return (invlink(eta + eps) - invlink(eta - eps)) / (2 * eps)

    except AttributeError:

        link_name = "identity"

        invlink = lambda x: x

        deriv = lambda x: np.ones_like(np.asarray(x), dtype=float)

    return link_name, invlink, deriv


@dataclass
class FactorSpec:
    kind: str                           # "cat" or "num"
    levels: Optional[List] = None       # for categorical
    depends_on: Optional[List[str]] = None  # raw variable deps for transformed factors


@dataclass
class FactorExtraction:
    factors: Dict[str, FactorSpec]
    name_map: Dict[str, str]
    deps_map: Dict[str, List[str]]


def _extract_factors(di, data: pd.DataFrame) -> FactorExtraction:
    factors: Dict[str, FactorSpec] = {}
    name_map: Dict[str, str] = {}
    deps_map: Dict[str, List[str]] = {}

    for fac, info in di.factor_infos.items():
        raw_name = fac.name() if hasattr(fac, "name") else str(fac)
        base = _base_name(raw_name)
        name_map[base] = raw_name

        state = getattr(info, "state", {}) or {}
        levels = [lv for lv in list(state.get("levels", [])) if pd.notna(lv)]

        if levels:
            # Patsy categorical with known levels
            factors[base] = FactorSpec(kind="cat", levels=levels, depends_on=[])
            continue

        if base in data.columns:
            s = data[base]
            if isinstance(s.dtype, CategoricalDtype):
                factors[base] = FactorSpec(kind="cat", levels=list(s.cat.categories), depends_on=[])
            elif not pd.api.types.is_numeric_dtype(s):
                factors[base] = FactorSpec(kind="cat", levels=list(pd.unique(s.dropna())), depends_on=[])
            else:
                factors[base] = FactorSpec(kind="num", levels=None, depends_on=[])
        else:
            # Transformed / EvalFactor (e.g., I(x**2), np.log(x), etc.)
            deps = _extract_dep_vars(raw_name, data.columns)
            factors[base] = FactorSpec(kind="num", levels=None, depends_on=deps)
            deps_map[base] = deps

    return FactorExtraction(factors=factors, name_map=name_map, deps_map=deps_map)


def _validate_at_levels(at: Dict[str, Union[float, str, List[Union[float, str]]]],
                        factors: Dict[str, FactorSpec]):
    for k, v in (at or {}).items():
        spec = factors[k]
        if spec.kind == "cat":
            vals = list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series)) else [v]
            bad = [x for x in vals if x not in (spec.levels or [])]
            if bad:
                raise ValueError(
                    f"at['{k}'] contains levels not in the model: {bad}. "
                    f"Allowed: {spec.levels}"
                )


def _determine_used_index_and_weights(result, data: pd.DataFrame) -> Tuple[Optional[pd.Index], Optional[pd.Series]]:
    """
    Attempts to recover the actual analysis sample index and (if present) model weights.
    """
    used_idx = None
    weights = None
    try:
        used_idx = getattr(result.model.data, "row_labels", None)
        if used_idx is None and len(getattr(result.model, "endog", [])) == len(data):
            used_idx = data.index
        if used_idx is not None:
            used_idx = pd.Index(used_idx).intersection(data.index)
    except Exception:
        used_idx = None

    try:
        weights = getattr(result.model, "weights", None)
        if weights is not None and used_idx is not None:
            weights = pd.Series(np.asarray(weights).reshape(-1), index=used_idx)
    except Exception:
        weights = None

    return used_idx, weights


def _build_reference_grid(
    data: pd.DataFrame,
    factors: Dict[str, FactorSpec],
    deps_map: Dict[str, List[str]],
    focal_set: set,
    at: Dict[str, Union[float, str, List[Union[float, str]]]],
    di,
    param_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Levels per factor (respect `at` first)
    grid_levels: Dict[str, List] = {}
    for name, meta in factors.items():
        if name in at:
            val = at[name]
            grid_levels[name] = list(val) if isinstance(val, (list, tuple, np.ndarray, pd.Series)) else [val]
        elif name in focal_set:
            if meta.kind == "cat":
                grid_levels[name] = meta.levels or []
            else:
                raise ValueError(
                    f"Numeric focal variable '{name}' requires explicit levels via at={{'{name}':[...]}}"
                )
        else:
            if meta.kind == "cat":
                grid_levels[name] = meta.levels or []
            else:
                if name in data.columns:
                    grid_levels[name] = [float(pd.to_numeric(data[name], errors="coerce").mean())]
                else:
                    # transformed numeric factor: inject raw dependencies later
                    pass

    if deps_map:
        needed = set(v for deps in deps_map.values() for v in deps)
        missing = [v for v in needed if v not in grid_levels]
        for v in missing:
            if v not in data.columns:
                continue
            s = data[v]
            if isinstance(s.dtype, CategoricalDtype):
                grid_levels[v] = list(s.cat.categories)
            elif not pd.api.types.is_numeric_dtype(s):
                grid_levels[v] = list(pd.unique(s.dropna()))
            else:
                grid_levels[v] = [float(pd.to_numeric(s, errors="coerce").mean())]

    # Cartesian grid
    keys = list(grid_levels.keys())
    values = [grid_levels[k] for k in keys]
    grid_raw = pd.DataFrame([dict(zip(keys, tup)) for tup in product(*values)])

    # Design matrix aligned to param order
    X_grid = patsy.build_design_matrices([di], grid_raw, return_type="dataframe")[0]
    Xg = X_grid.reindex(columns=param_names, fill_value=0.0)
    return grid_raw, Xg


def _apply_marginal_weights(
    grid_raw: pd.DataFrame,
    data: pd.DataFrame,
    used_idx: Optional[pd.Index],
    weight: str,
    focal_set: set,
    at: Dict[str, Union[float, str, List[Union[float, str]]]],
    factors: Dict[str, FactorSpec],
    drop_unseen: bool
) -> pd.DataFrame:
    grid_raw = grid_raw.copy()
    grid_raw["_weight"] = 1.0

    if weight == "proportional":
        nonfocal_cats = [
            nm for nm, m in factors.items()
            if (nm not in focal_set) and (nm not in at) and (m.kind == "cat")
        ]
        if nonfocal_cats:
            source_df = data.loc[used_idx] if used_idx is not None else data
            # Avoid pandas warning by passing a single column name instead of a list
            grouper = nonfocal_cats[0] if len(nonfocal_cats) == 1 else nonfocal_cats
            grp = _groupby(source_df[nonfocal_cats], grouper, dropna=False).size()
            total = grp.sum()
            joint_props = (grp / total).to_dict()

            def _row_prop(row):
                key = tuple(row[c] for c in nonfocal_cats)
                return float(joint_props.get(key, 0.0))

            grid_raw["_weight"] = grid_raw.apply(
                (lambda r: _row_prop(r) if nonfocal_cats else 1.0), axis=1
            )

            if drop_unseen:
                grid_raw = grid_raw.loc[grid_raw["_weight"] > 0].copy()

            if np.allclose(grid_raw["_weight"].values.sum(), 0.0):
                grid_raw["_weight"] = 1.0

    return grid_raw


def _normalize_weights_by_cell(grid_raw: pd.DataFrame, cell_keys: List[str]) -> pd.DataFrame:
    grid_raw = grid_raw.copy()
    if len(cell_keys) > 0:
        sums = _groupby(grid_raw, cell_keys, sort=False)["_weight"].transform("sum")
        with np.errstate(divide="ignore", invalid="ignore"):
            grid_raw["_weight_norm_by_cell"] = np.where(sums > 0, grid_raw["_weight"] / sums, 0.0)
    else:
        grid_raw["_weight_norm_by_cell"] = 1.0
    return grid_raw


def _label_for_spec_tuple(specs: List[str], tup: Tuple) -> str:
    if len(specs) == 1:
        return str(tup[0])
    parts = [f"{nm}={val}" for nm, val in zip(specs, tup)]
    return " × ".join(parts)


# ---------------------- Core computational blocks ----------------------

def _compute_emms_per_cell(
    grid_raw: pd.DataFrame,
    Xg: pd.DataFrame,
    beta: np.ndarray,
    V: np.ndarray,
    invlink: Callable,
    deriv: Callable,
    transform: str,
    level: float,
    df_global: float,
    specs: List[str],
    by: List[str],
) -> Tuple[pd.DataFrame, Dict[Tuple, List[Tuple[Tuple, np.ndarray, float, float]]]]:
    cell_keys = specs + by
    # Avoid pandas FutureWarning by not passing a list of length 1 as the grouper
    if cell_keys:
        grouper = cell_keys[0] if len(cell_keys) == 1 else cell_keys
        grouped = _groupby(grid_raw, grouper, sort=False)
    else:
        grouped = [((), grid_raw)]

    emm_rows = []
    by_slices: Dict[Tuple, List[Tuple[Tuple, np.ndarray, float, float]]] = {}

    for key_vals, subgrid in grouped:
        if subgrid.shape[0] == 0:
            continue
        w = subgrid["_weight"].values.astype(float)
        if w.size == 0:
            continue
        total_w = w.sum()
        w = (w / total_w) if total_w > 0 else np.full_like(w, 1.0 / float(w.size))

        L = (Xg.loc[subgrid.index].values.T @ w).reshape(1, -1)

        eta = float((L @ beta).item())
        var_eta = float((L @ V @ L.T).item())
        se_eta = float(np.sqrt(max(var_eta, 0.0)))

        if transform == "response":
            mu = float(invlink(eta))
            se_mu = float(abs(deriv(eta)) * se_eta)
        elif transform == "link":
            mu, se_mu = eta, se_eta
        else:
            raise ValueError("transform must be 'response' or 'link'")

        df_use = df_global
        crit = norm.ppf(0.5 + level / 2) if np.isinf(df_use) else tdist.ppf(0.5 + level / 2, df_use)

        lo_eta, hi_eta = eta - crit * se_eta, eta + crit * se_eta
        if transform == "response":
            lo, hi = float(invlink(lo_eta)), float(invlink(hi_eta))
        else:
            lo, hi = float(lo_eta), float(hi_eta)

        row = {}
        if cell_keys:
            if not isinstance(key_vals, tuple):
                key_vals = (key_vals,)
            for k, v in zip(cell_keys, key_vals):
                row[k] = v

        row.update({
            "emmean": mu,
            "SE": se_mu,
            "df": df_use,
            "lower.CL": lo,
            "upper.CL": hi
        })
        emm_rows.append(row)

        if by:
            by_vals = tuple(row[bk] for bk in by)
            spec_vals = tuple(row[sk] for sk in specs)
        else:
            by_vals = ()
            spec_vals = tuple(row[sk] for sk in specs) if specs else ("grand",)

        by_slices.setdefault(by_vals, []).append((spec_vals, L, eta, se_eta))

    return pd.DataFrame(emm_rows), by_slices


def _pairwise_contrasts_for_slice(
    entries,
    specs,
    by_vals,
    beta,
    V,
    invlink,
    deriv,
    link_name,
    contrast_transform,
    df_global,
    df_provider,
    by,
):
    rows = []

    spec_keys = [e[0] for e in entries]
    L_rows = [e[1] for e in entries]
    k_groups = len(entries)

    if k_groups < 2:
        return rows

    for i, j in combinations(range(k_groups), 2):

        name_i = _label_for_spec_tuple(specs, spec_keys[i])
        name_j = _label_for_spec_tuple(specs, spec_keys[j])

        Li = L_rows[i]
        Lj = L_rows[j]
        Lc = Li - Lj

        # ---------------------------
        # LINK SCALE (R emmeans default)
        # ---------------------------
        delta = float((Lc @ beta).item())
        var   = float((Lc @ V @ Lc.T).item())
        se    = float(np.sqrt(max(var, 0.0)))

        df_c = df_global
        if df_provider is not None:
            try:
                tmp = df_provider(Lc)
                if tmp is not None and np.isfinite(tmp):
                    df_c = float(tmp)
            except Exception:
                pass

        stat = delta / se if se > 0 else np.nan

        if np.isinf(df_c):
            p = 2 * (1 - norm.cdf(abs(stat))) if np.isfinite(stat) else np.nan
        else:
            p = 2 * (1 - tdist.cdf(abs(stat), df_c)) if np.isfinite(stat) else np.nan

        row = {
            "contrast": f"{name_i} - {name_j}",
            "ratio": np.exp(delta),
            "SE": se,
            "df": df_c,
            "stat": stat,
            "p.value": p,
            "__i": i,
            "__j": j,
        }

        for idx, bname in enumerate(by):
            row[bname] = by_vals[idx]

        rows.append(row)

    return rows


def _custom_contrasts_for_slice(
    entries,
    contrasts,
    specs,
    by_vals,
    beta,
    V,
    invlink,
    deriv,
    contrast_transform,
    link_name,
    df_global,
    df_provider,
    by,
):
    rows = []

    L_rows = [e[1] for e in entries]
    k_groups = len(entries)

    if k_groups == 0:
        return rows

    for label, wvec in contrasts:

        w = np.asarray(wvec, dtype=float).reshape(-1)

        if w.shape[0] != k_groups:
            raise ValueError(
                f"Contrast '{label}' length {w.shape[0]} != {k_groups}"
            )

        # Linear combination on LINK SCALE
        Lc = sum(w[g] * L_rows[g] for g in range(k_groups))

        delta = float((Lc @ beta).item())
        var   = float((Lc @ V @ Lc.T).item())
        se    = float(np.sqrt(max(var, 0.0)))

        df_c = df_global
        if df_provider is not None:
            try:
                tmp = df_provider(Lc)
                if tmp is not None and np.isfinite(tmp):
                    df_c = float(tmp)
            except Exception:
                pass

        stat = delta / se if se > 0 else np.nan

        if np.isinf(df_c):
            p = 2 * (1 - norm.cdf(abs(stat))) if np.isfinite(stat) else np.nan
        else:
            p = 2 * (1 - tdist.cdf(abs(stat), df_c)) if np.isfinite(stat) else np.nan

        row = {
            "contrast": str(label),
            "estimate": delta,
            "SE": se,
            "df": df_c,
            "stat": stat,
            "p.value": p,
        }

        for idx, bname in enumerate(by):
            row[bname] = by_vals[idx]

        rows.append(row)

    return rows


def _adjust_pvalues_per_slice(
    contrasts_df: pd.DataFrame,
    method: str,
    result,
    data: pd.DataFrame,
    specs: List[str],
    by: List[str],
    used_idx: Optional[pd.Index],
    model_weights: Optional[pd.Series],
) -> pd.Series:
    """
    Returns a Series aligned to contrasts_df.index with adjusted p-values.
    Implements strict Tukey/Tukey–Kramer when valid; otherwise falls back to
    Holm/Bonferroni/Sidak/FDR as requested (with 'tukey' aliasing to Holm).
    """
    if by:
        # Avoid passing a single-element list as the grouper to avoid pandas warnings
        grouper = by[0] if isinstance(by, list) and len(by) == 1 else by
        slice_iter = contrasts_df.groupby(grouper, dropna=False, sort=False)
    else:
        slice_iter = [((), contrasts_df)]

    # conditions for exact Tukey:
    is_ols = isinstance(getattr(result, "model", None), sm.OLS)
    strict_oneway = is_ols and (method == "tukey") and (len(specs) == 1)
    no_weights = (model_weights is None)

    use_real_tukey_global = (method == "tukey" and _HAVE_PSTURNG and strict_oneway and no_weights)

    p_adj_chunks = []
    for by_vals, subdf in slice_iter:
        if use_real_tukey_global:
            factor = specs[0]
            if used_idx is None:
                p_adj_chunks.append(_holm_chunk(subdf))
                continue

            # filter analysis rows by BY levels if any
            if by:
                mask = pd.Series(True, index=used_idx)
                for bname, bval in zip(by, by_vals if isinstance(by_vals, tuple) else (by_vals,)):
                    mask &= (data.loc[used_idx, bname] == bval)
                used_in_slice = used_idx[mask]
            else:
                used_in_slice = used_idx

            data_slice = data.loc[used_in_slice]

            if factor not in data_slice.columns:
                p_adj_chunks.append(_holm_chunk(subdf))
                continue

            # group sizes per level
            # Use single-column grouper when the factor list has length 1 to avoid pandas warnings
            level_counts = (
                _groupby(data_slice[[factor]], factor, dropna=False)
                .size().astype(float)
            )

            try:
                mse = float(result.mse_resid)
                df_tukey = int(result.df_resid)
            except Exception:
                p_adj_chunks.append(_holm_chunk(subdf))
                continue

            # Tukey–Kramer q statistics using LINK-scale deltas
            q_vals = []
            for _, r in subdf.iterrows():

                key_i = r["__level_i"] if "__level_i" in r else r["__name_i"]
                key_j = r["__level_j"] if "__level_j" in r else r["__name_j"]

                ni = float(level_counts.get(key_i, np.nan))
                nj = float(level_counts.get(key_j, np.nan))

                if not np.isfinite(ni) or not np.isfinite(nj) or ni <= 1 or nj <= 1:
                    q_vals.append(np.nan)
                    continue

                # ALWAYS use final model contrast estimate
                delta = float(r["estimate"])

                denom = np.sqrt(mse * 0.5 * (1.0/ni + 1.0/nj))

                q_vals.append(abs(delta) / denom if denom > 0 else np.nan)

            k = int(level_counts.shape[0])
            padj = [1 - psturng(q, k, df_tukey) if np.isfinite(q) else np.nan for q in q_vals]
            p_adj_chunks.append(pd.Series(padj, index=subdf.index))
        else:
            # General models or 'tukey' fallback → Holm
            eff = "holm" if method == "tukey" else method
            p = subdf["p.value"].to_numpy(dtype=float)
            mask = np.isfinite(p)
            p_adj = np.full_like(p, np.nan, dtype=float)
            if mask.any():
                _, padj_valid, _, _ = multipletests(p[mask], method=eff)
                p_adj[mask] = padj_valid
            p_adj_chunks.append(pd.Series(p_adj, index=subdf.index))

    return pd.concat(p_adj_chunks).sort_index()


def _holm_chunk(subdf: pd.DataFrame) -> pd.Series:
    p = subdf["p.value"].to_numpy(dtype=float)
    mask = np.isfinite(p)
    p_adj = np.full_like(p, np.nan, dtype=float)
    if mask.any():
        _, padj_valid, _, _ = multipletests(p[mask], method="holm")
        p_adj[mask] = padj_valid
    return pd.Series(p_adj, index=subdf.index)


def _finalize_contrasts_df(df):

    out = df.copy()

    df_vals = out["df"].astype(float).to_numpy()
    stat_vals = out["stat"].astype(float).to_numpy()

    out["t.ratio"] = np.where(np.isfinite(df_vals), stat_vals, np.nan)
    out["z.ratio"] = np.where(~np.isfinite(df_vals), stat_vals, np.nan)

    if "__contrast_transform" in out.columns:

        transform = out["__contrast_transform"].iloc[0]
        link = out["__link_name"].iloc[0]

        if transform == "response":

            if link == "log":

                out["ratio"] = np.exp(out["estimate"])

                out["contrast"] = out["contrast"].str.replace(" - ", " / ")

                out.drop(columns="estimate", inplace=True)

            elif link == "identity":
                pass

            else:
                out["estimate"] = np.vectorize(
                    lambda x: x
                )(out["estimate"])

    out.drop(
        columns=[
            "__contrast_transform",
            "__link_name",
            # "__delta_link",
            "__i",
            "__j",
            "__name_i",
            "__name_j",
            "__level_i",
            "__level_j",
        ],
        errors="ignore",
        inplace=True,
    )

    return out

def make_example_data(seed: int = 42, n: int = 400) -> pd.DataFrame:
    """
    Simulated dataset for demonstrating estimated marginal means (EMMs).

    Design:
    - Factor A: treatment (3 levels)
    - Factor B: condition (2 levels)
    - Covariate: age (continuous)
    - Random noise outcome with interaction effects
    """

    rng = np.random.default_rng(seed)

    # Factors
    A = rng.choice(["control", "drugA", "drugB"], size=n)
    B = rng.choice(["low_load", "high_load"], size=n)

    # Continuous covariate
    age = rng.normal(35, 10, size=n)

    # Encode effects (true underlying model)
    A_effect = {"control": 0.0, "drugA": 1.2, "drugB": 1.8}
    B_effect = {"low_load": 0.0, "high_load": -0.8}

    # Interaction: drug effects stronger under high load
    interaction = {
        ("drugA", "high_load"): 0.6,
        ("drugB", "high_load"): 1.0,
        ("control", "high_load"): 0.0,
        ("drugA", "low_load"): 0.0,
        ("drugB", "low_load"): 0.0,
        ("control", "low_load"): 0.0,
    }

    # Construct linear signal
    y = np.zeros(n)

    for i in range(n):
        base = 5.0
        y[i] = (
            base
            + A_effect[A[i]]
            + B_effect[B[i]]
            + interaction[(A[i], B[i])]
            + 0.02 * (age[i] - 35)   # small covariate effect
            + rng.normal(0, 1.0)    # noise
        )

    df = pd.DataFrame({
        "y": y,
        "treatment": A,
        "load": B,
        "age": age
    })

    # Make categorical (important for emmeans-style usage)
    df["treatment"] = df["treatment"].astype("category")
    df["load"] = df["load"].astype("category")

    return df


if __name__ == "__main__":
    df = make_example_data()
    print(df.head())