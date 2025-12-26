"""HiGHS JavaScript/WebAssembly solver bridge for Pyodide.

This module provides integration with the highs-js solver for use in
browser environments via Pyodide.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from linopy import Model

logger = logging.getLogger(__name__)


def solve_with_highs_js(model: Model, **kwargs):
    """Solve optimization model using HiGHS JavaScript/WebAssembly solver.

    This function extracts the linear programming problem from the linopy model,
    converts it to HiGHS JSON format, calls the JavaScript HiGHS solver,
    and converts the solution back.

    Parameters
    ----------
    model : linopy.Model
        The optimization model to solve
    **kwargs
        Additional solver options (currently ignored)

    Returns
    -------
    status : str
        Solver status ("ok" or error status)
    condition : str
        Termination condition ("optimal", "infeasible", etc.)

    Raises
    ------
    ImportError
        If not running in Pyodide or if js_highs_solve is not available
    """
    try:
        import js
    except ImportError as e:
        msg = f"HiGHS-JS solver requires Pyodide environment: {e}"
        logger.error(msg)
        raise ImportError(msg) from e

    logger.info("Extracting model for HiGHS-JS solver...")

    # Get constraint matrix and bounds
    A, b_lower, b_upper, c, sense = _extract_model_data(model)

    # Get variable bounds
    v_lower, v_upper = _extract_variable_bounds(model)

    # Build HiGHS problem dictionary
    problem = _build_highs_problem(A, b_lower, b_upper, c, v_lower, v_upper, sense)

    logger.info(
        f"Calling HiGHS-JS solver with {len(problem['cols'])} variables "
        f"and {len(problem['rows'])} constraints..."
    )

    # Call JavaScript HiGHS solver
    try:
        result = js.js_highs_solve(problem)
    except Exception as e:
        msg = f"HiGHS-JS solver failed: {e}"
        logger.error(msg)
        return "error", "unknown"

    # Parse results
    status, condition = _parse_highs_result(result, model)

    return status, condition


def _extract_model_data(model: Model):
    """Extract constraint matrix and objective from linopy model."""
    import scipy.sparse as sp

    # Get objective coefficients
    obj_const, obj_coeffs = model.objective.to_matrix()
    c = obj_coeffs

    # Get constraints
    A, b = model.constraints.to_matrix()

    # Get constraint signs and build bounds
    signs = model.constraints.sign.to_series()
    rhs = b

    b_lower = np.full_like(rhs, -np.inf)
    b_upper = np.full_like(rhs, np.inf)

    for i, (idx, sign) in enumerate(signs.items()):
        if sign == "==":
            b_lower[i] = rhs[i]
            b_upper[i] = rhs[i]
        elif sign == ">=":
            b_lower[i] = rhs[i]
        elif sign == "<=":
            b_upper[i] = rhs[i]

    # Get objective sense
    sense = "minimize" if model.objective.sense == "min" else "maximize"

    return A, b_lower, b_upper, c, sense


def _extract_variable_bounds(model: Model):
    """Extract variable bounds from linopy model."""
    # Get bounds from matrices
    lower = model.matrices.lb
    upper = model.matrices.ub

    # Replace NaN with inf
    v_lower = np.where(np.isnan(lower), -np.inf, lower)
    v_upper = np.where(np.isnan(upper), np.inf, upper)

    return v_lower, v_upper


def _build_highs_problem(A, b_lower, b_upper, c, v_lower, v_upper, sense):
    """Build HiGHS JSON problem format.

    The HiGHS-JS API expects a problem dictionary with:
    - sense: "minimize" or "maximize"
    - offset: constant offset in objective
    - cols: list of column (variable) definitions
    - rows: list of row (constraint) definitions
    """
    problem = {
        "sense": sense,
        "offset": 0.0,
        "cols": [],
        "rows": [],
    }

    # Add variables (columns)
    for i in range(len(c)):
        col = {
            "name": f"x{i}",
            "obj": float(c[i]) if not np.isnan(c[i]) else 0.0,
        }
        if not np.isinf(v_lower[i]):
            col["lb"] = float(v_lower[i])
        if not np.isinf(v_upper[i]):
            col["ub"] = float(v_upper[i])
        problem["cols"].append(col)

    # Add constraints (rows)
    # Convert to CSR format for efficient row access
    if not isinstance(A, (np.ndarray,)):
        import scipy.sparse as sp
        if sp.issparse(A):
            A = A.tocsr()
        else:
            A = np.asarray(A)

    for i in range(A.shape[0]):
        row = {"coeffs": []}

        # Get row data
        if hasattr(A, 'getrow'):  # sparse matrix
            row_data = A.getrow(i)
            for j, val in zip(row_data.indices, row_data.data):
                if val != 0:
                    row["coeffs"].append({"col": int(j), "val": float(val)})
        else:  # dense array
            row_vals = A[i, :]
            for j, val in enumerate(row_vals):
                if val != 0:
                    row["coeffs"].append({"col": int(j), "val": float(val)})

        if not np.isinf(b_lower[i]):
            row["lb"] = float(b_lower[i])
        if not np.isinf(b_upper[i]):
            row["ub"] = float(b_upper[i])

        problem["rows"].append(row)

    return problem


def _parse_highs_result(result, model: Model):
    """Parse HiGHS result and update model solution.

    Parameters
    ----------
    result : dict
        Result dictionary from HiGHS-JS containing:
        - Status: solver status string
        - ObjectiveValue: optimal objective value
        - Columns: list of column results with Primal values
        - Rows: list of row results with Dual values
    model : Model
        Linopy model to update with solution

    Returns
    -------
    status : str
        Solver status ("ok", "warning", or "error")
    condition : str
        Termination condition ("optimal", "infeasible", etc.)
    """
    # Check status
    status_map = {
        "Optimal": ("ok", "optimal"),
        "Infeasible": ("warning", "infeasible"),
        "Unbounded": ("warning", "unbounded"),
        "UnboundedOrInfeasible": ("warning", "infeasible_or_unbounded"),
    }

    result_status = result.get("Status", "Unknown")
    status, condition = status_map.get(result_status, ("error", "unknown"))

    if status != "ok":
        logger.warning(f"HiGHS-JS solver finished with status: {result_status}")
        return status, condition

    # Extract solution values
    columns = result.get("Columns", [])
    solution_values = np.array([col.get("Primal", 0.0) for col in columns])

    # Extract dual values
    rows = result.get("Rows", [])
    dual_values = np.array([row.get("Dual", 0.0) for row in rows])

    # Create solution series with proper labels
    solution = pd.Series(solution_values, index=model.matrices.vlabels)
    dual = pd.Series(dual_values, index=model.matrices.clabels)

    # Store solution in model
    model.solution = solution
    model.dual = dual

    # Set objective value if available
    if "ObjectiveValue" in result:
        model.objective_value = result["ObjectiveValue"]
        logger.info(f"HiGHS-JS solver completed successfully. Objective: {model.objective_value}")
    else:
        logger.info("HiGHS-JS solver completed successfully.")

    return status, condition
