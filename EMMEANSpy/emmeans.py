# Import EMM dependencies
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union, List, Optional, Dict, Tuple
from unittest import result

import numpy as np
import pandas as pd
from scipy.stats import norm, t as tdist
from statsmodels.stats.multitest import multipletests

try:
    # statsmodels >= 0.14
    from statsmodels.stats.libqsturng import psturng
    _HAVE_PSTURNG = True
except Exception:
    _HAVE_PSTURNG = False

from .utils import (
    _get_model_params_and_vcov,
    _get_link_and_deriv,
    _extract_factors,
    _validate_at_levels,
    _build_reference_grid,
    _determine_used_index_and_weights,
    _apply_marginal_weights,
    _normalize_weights_by_cell,
    _compute_emms_per_cell,
    _get_design_info_or_raise,
    _pairwise_contrasts_for_slice,
    _custom_contrasts_for_slice,
    _adjust_pvalues_per_slice,
    _finalize_contrasts_df
)
# ------------------------------ Public API ------------------------------

def emmeans(
    result,
    data: pd.DataFrame,
    specs: Union[str, List[str]],
    *,
    by: Optional[Union[str, List[str]]] = None,
    at: Optional[Dict[str, Union[float, str, List[Union[float, str]]]]] = None,
    weight: str = "equal",
    transform: str = "response",
    level: float = 0.95,
    contrasts: Union[str, List[Tuple[str, np.ndarray]]] = "pairwise",
    contrast_transform: str = "link",
    adjust: Optional[str] = "tukey",
    df_method: str = "resid",
    vcov: Optional[np.ndarray] = None,
    return_grid: bool = False,
    # ---- extended knobs ----
    df_provider: Optional[Callable[[np.ndarray], float]] = None,
    drop_unseen: bool = False,
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Compute estimated marginal means (EMMs) and (within-BY) contrasts for statsmodels results.

    Refactored for clarity/DRY while preserving computations and outputs.
    """
    # ---------------- A. Parse/validate inputs ----------------
    specs = [specs] if isinstance(specs, str) else list(specs)
    by = [by] if isinstance(by, str) and by is not None else (list(by) if by else [])

    if weight not in {"equal", "proportional"}:
        raise ValueError("weight must be 'equal' or 'proportional'")

    at = at or {}
    focal_set = set(specs) | set(by)

    # beta, param_names, V = _get_model_params_and_vcov(result, vcov)
    # invlink, deriv = _get_link_and_deriv(result)
    beta, param_names, V = _get_model_params_and_vcov(result, vcov)
    link_name, invlink, deriv = _get_link_and_deriv(result)

    if contrast_transform not in {"link", "response"}:
        raise ValueError("contrast_transform must be 'link' or 'response'")

    di = _get_design_info_or_raise(result)

    # Degrees of freedom for EMM CIs (global default; per-contrast handled with df_provider)
    df_global = np.inf if df_method == "wald" else getattr(result, "df_resid", np.inf)

    # ---------------- B. Extract factors & validate 'at' ----------------
    fx = _extract_factors(di, data)
    for nm in (set(specs) | set(by) | set(at.keys())):
        if nm not in fx.factors:
            raise ValueError(
                f"'{nm}' is not a recognized model factor. "
                f"Known factors (by base name): {list(fx.factors.keys())}"
            )
    _validate_at_levels(at, fx.factors)

    # ---------------- C. Build reference grid & design ----------------
    grid_raw, Xg = _build_reference_grid(
        data=data,
        factors=fx.factors,
        deps_map=fx.deps_map,
        focal_set=focal_set,
        at=at,
        di=di,
        param_names=param_names,
    )

    # ---------------- D. Analysis index/weights for Tukey conditions ----------------
    used_idx, model_weights = _determine_used_index_and_weights(result, data)

    # ---------------- E. Marginalization weights ----------------
    grid_raw = _apply_marginal_weights(
        grid_raw=grid_raw,
        data=data,
        used_idx=used_idx,
        weight=weight,
        focal_set=focal_set,
        at=at,
        factors=fx.factors,
        drop_unseen=drop_unseen
    )
    # Keep Xg aligned to current grid rows if we dropped unseen
    Xg = Xg.loc[grid_raw.index]

    cell_keys = specs + by
    grid_raw = _normalize_weights_by_cell(grid_raw, cell_keys)

    # ---------------- F. EMMs ----------------
    emm_df, by_slices = _compute_emms_per_cell(
        grid_raw=grid_raw,
        Xg=Xg,
        beta=beta,
        V=V,
        invlink=invlink,
        deriv=deriv,
        transform=transform,
        level=level,
        df_global=df_global,
        specs=specs,
        by=by
    )

    # ---------------- G. Contrasts ----------------

    contrasts_df = None
    if (contrasts == "pairwise" or isinstance(contrasts, list)) and len(emm_df) > 0 and len(specs) > 0:
        all_rows = []
        for by_vals, entries in by_slices.items():
            if contrasts == "pairwise":
                all_rows.extend(
                    _pairwise_contrasts_for_slice(
                        entries=entries,
                        specs=specs,
                        by_vals=by_vals,
                        beta=beta,
                        V=V,
                        link_name=link_name,
                        invlink=invlink,
                        deriv=deriv,
                        contrast_transform=contrast_transform,
                        df_global=df_global,
                        df_provider=df_provider,
                        by=by
                    )
                )
            else:
                all_rows.extend(
                    _custom_contrasts_for_slice(
                        entries=entries,
                        contrasts=contrasts,  # type: ignore[arg-type]
                        specs=specs,
                        by_vals=by_vals,
                        beta=beta,
                        V=V,
                        link_name=link_name,
                        invlink=invlink,
                        deriv=deriv,
                        contrast_transform=contrast_transform,
                        df_global=df_global,
                        df_provider=df_provider,
                        by=by
                    )
                )
        if all_rows:
            contrasts_df = pd.DataFrame(all_rows)

            if adjust:
                method = (adjust or "").lower()
                contrasts_df["p.value.adj"] = _adjust_pvalues_per_slice(
                    contrasts_df=contrasts_df,
                    method=method,
                    result=result,
                    data=data,
                    specs=specs,
                    by=by,
                    used_idx=used_idx,
                    model_weights=model_weights,
                )

            contrasts_df = _finalize_contrasts_df(contrasts_df)

    out: Dict[str, Optional[pd.DataFrame]] = {"emm": emm_df, "contrasts": contrasts_df}
    if return_grid:
        out["grid"] = grid_raw
    return out