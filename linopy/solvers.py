#!/usr/bin/env python3
"""
Linopy module for solving lp files with different solvers.
"""

from __future__ import annotations

import contextlib
import enum
import io
import logging
import os
import re
import sys
import warnings
from abc import ABC, abstractmethod
from collections import namedtuple
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np
import pandas as pd
from packaging.version import parse as parse_version

from linopy.constants import (
    Result,
    Solution,
    SolverStatus,
    Status,
    TerminationCondition,
)
from linopy.solver_capabilities import (
    SolverFeature,
    get_solvers_with_feature,
)

if TYPE_CHECKING:
    import gurobipy

    from linopy.model import Model

EnvType = TypeVar("EnvType")

# Generated from solver_capabilities registry for backward compatibility


FILE_IO_APIS = ["lp", "lp-polars", "mps"]
IO_APIS = FILE_IO_APIS + ["direct"]

available_solvers = []

# Auto-detect available solvers
# HiGHS-JS (for Pyodide/WebAssembly environments)
try:
    import js
    if hasattr(js, 'js_highs_solve'):
        available_solvers.append("highs-js")
except (ImportError, AttributeError):
    pass

_new_highspy_mps_layout = True

logger = logging.getLogger(__name__)


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

    # Check if js_highs_solve is available
    if not hasattr(js, 'js_highs_solve'):
        msg = (
            "HiGHS-JS bridge function 'js_highs_solve' not found. "
            "Please ensure the HiGHS-JS library is loaded and the bridge function is exposed to Python. "
            "See PYODIDE_USAGE.md for setup instructions."
        )
        logger.error(msg)
        raise ImportError(msg)

    logger.info("Extracting model for HiGHS-JS solver...")

    # Get constraint matrix and bounds
    try:
        A, b_lower, b_upper, c, sense, offset = _extract_model_data(model)
    except Exception as e:
        msg = f"Failed to extract model data: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # Get variable bounds and types
    try:
        v_lower, v_upper, vtypes = _extract_variable_bounds(model)
    except Exception as e:
        msg = f"Failed to extract variable bounds: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    # Validate model is not empty
    if len(c) == 0:
        msg = "Model has no variables. Cannot solve empty model."
        logger.error(msg)
        raise ValueError(msg)

    # Build HiGHS problem dictionary
    try:
        problem = _build_highs_problem(A, b_lower, b_upper, c, v_lower, v_upper, vtypes, sense, offset)
    except Exception as e:
        msg = f"Failed to build HiGHS problem: {e}"
        logger.error(msg)
        raise ValueError(msg) from e

    logger.info(
        f"Calling HiGHS-JS solver with {len(problem['cols'])} variables "
        f"and {len(problem['rows'])} constraints (model type: {model.type})..."
    )

    # Call JavaScript HiGHS solver
    try:
        result = js.js_highs_solve(problem)
    except AttributeError as e:
        msg = (
            f"Failed to call js_highs_solve: {e}. "
            "Ensure the JavaScript bridge function is properly configured."
        )
        logger.error(msg)
        return "error", "unknown"
    except Exception as e:
        msg = f"HiGHS-JS solver execution failed: {e}"
        logger.error(msg)
        return "error", "unknown"

    # Parse results
    try:
        status, condition = _parse_highs_result(result, model)
    except Exception as e:
        msg = f"Failed to parse solver result: {e}"
        logger.error(msg)
        return "error", "unknown"

    return status, condition


def _extract_model_data(model: Model):
    """Extract constraint matrix and objective from linopy model."""
    import scipy.sparse as sp

    # Get objective coefficients and constant
    obj_const, obj_coeffs = model.objective.to_matrix()
    c = obj_coeffs
    offset = float(obj_const) if obj_const is not None else 0.0

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

    return A, b_lower, b_upper, c, sense, offset


def _extract_variable_bounds(model: Model):
    """Extract variable bounds and types from linopy model."""
    # Get bounds from matrices
    lower = model.matrices.lb
    upper = model.matrices.ub

    # Replace NaN with inf
    v_lower = np.where(np.isnan(lower), -np.inf, lower)
    v_upper = np.where(np.isnan(upper), np.inf, upper)

    # Get variable types (continuous, integer, binary)
    # vtypes will be an array with values like 0 (continuous), 1 (integer), 2 (binary)
    if hasattr(model.matrices, 'vtypes'):
        vtypes = model.matrices.vtypes
    else:
        # If vtypes not available, assume all continuous
        vtypes = np.zeros(len(lower), dtype=int)

    return v_lower, v_upper, vtypes


def _build_highs_problem(A, b_lower, b_upper, c, v_lower, v_upper, vtypes, sense, offset):
    """Build HiGHS JSON problem format.

    The HiGHS-JS API expects a problem dictionary with:
    - sense: "minimize" or "maximize"
    - offset: constant offset in objective
    - cols: list of column (variable) definitions
    - rows: list of row (constraint) definitions
    """
    problem = {
        "sense": sense,
        "offset": offset,
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

        # Add variable type if integer or binary
        # vtypes: 0 = continuous, 1 = integer, 2 = binary
        if vtypes[i] == 1:
            col["type"] = "integer"
        elif vtypes[i] == 2:
            col["type"] = "binary"
        # continuous is the default, no need to specify

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
        Termination condition ("optimal", "infeasible", etc.")
    """
    # Validate result is a dictionary
    if not isinstance(result, dict):
        logger.error(f"Expected dict from HiGHS-JS, got {type(result)}")
        return "error", "unknown"

    # Check status
    status_map = {
        "Optimal": ("ok", "optimal"),
        "Infeasible": ("warning", "infeasible"),
        "Unbounded": ("warning", "unbounded"),
        "UnboundedOrInfeasible": ("warning", "infeasible_or_unbounded"),
        "TimeLimit": ("warning", "time_limit"),
        "IterationLimit": ("warning", "iteration_limit"),
    }

    result_status = result.get("Status", "Unknown")
    status, condition = status_map.get(result_status, ("error", "unknown"))

    if status != "ok":
        logger.warning(f"HiGHS-JS solver finished with status: {result_status}")
        # Still try to extract partial solution if available
        if status == "warning" and "Columns" in result:
            logger.info("Attempting to extract partial solution...")
        else:
            return status, condition

    # Extract solution values
    columns = result.get("Columns", [])
    if not columns:
        logger.warning("No solution columns returned from solver")
        return status, condition

    # Validate column count matches expected
    expected_vars = len(model.matrices.vlabels)
    if len(columns) != expected_vars:
        logger.warning(
            f"Solution has {len(columns)} variables but model has {expected_vars}. "
            "This may indicate a problem with the solver or model extraction."
        )

    solution_values = np.array([col.get("Primal", 0.0) for col in columns])

    # Extract dual values
    rows = result.get("Rows", [])
    if rows:
        expected_cons = len(model.matrices.clabels)
        if len(rows) != expected_cons:
            logger.warning(
                f"Solution has {len(rows)} constraints but model has {expected_cons}"
            )
        dual_values = np.array([row.get("Dual", 0.0) for row in rows])
    else:
        # For MILPs, dual values might not be available
        logger.info("No dual values returned (this is expected for MILP problems)")
        dual_values = np.zeros(len(model.matrices.clabels))

    # Create solution series with proper labels
    solution = pd.Series(solution_values, index=model.matrices.vlabels)
    dual = pd.Series(dual_values, index=model.matrices.clabels)

    # Store solution in model
    model.solution = solution
    model.dual = dual

    # Set objective value if available
    if "ObjectiveValue" in result:
        model.objective_value = result["ObjectiveValue"]
        logger.info(f"HiGHS-JS solver completed successfully. Objective: {model.objective_value:.6f}")
    else:
        logger.warning("No objective value returned from solver")
        model.objective_value = np.nan

    return status, condition


# using enum to match solver subclasses with names
class SolverName(enum.Enum):
    HighsJS = 'highs-js'


def path_to_string(path: Path) -> str:
    """
    Convert a pathlib.Path to a string.
    """
    return str(path.resolve())


def read_sense_from_problem_file(problem_fn: Path | str) -> str:
    with open(problem_fn) as file:
        f = file.read()
    file_format = read_io_api_from_problem_file(problem_fn)
    if file_format == "lp":
        return "min" if "min" in f.lower() else "max"
    elif file_format == "mps":
        return "max" if "OBJSENSE\n  MAX\n" in f else "min"
    else:
        msg = "Unsupported problem file format."
        raise ValueError(msg)


def read_io_api_from_problem_file(problem_fn: Path | str) -> str:
    if isinstance(problem_fn, Path):
        return problem_fn.suffix[1:]
    else:
        return problem_fn.split(".")[-1]


def maybe_adjust_objective_sign(
    solution: Solution, io_api: str | None, sense: str | None
) -> Solution:
    if sense == "min":
        return solution
    if np.isnan(solution.objective):
        return solution
    if io_api == "mps" and not _new_highspy_mps_layout:
        logger.info(
            "Adjusting objective sign due to switched coefficients in MPS file."
        )
        solution.objective *= -1
    return solution


class Solver(ABC, Generic[EnvType]):
    """
    Abstract base class for solving a given linear problem.

    All relevant functions are passed on to the specific solver subclasses.
    Subclasses must implement the `solve_problem_from_model()` and
    `solve_problem_from_file()` methods.
    """

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        self.solver_options = solver_options

        # Check for the solver to be initialized whether the package is installed or not.
        if self.solver_name.value not in available_solvers:
            msg = f"Solver package for '{self.solver_name.value}' is not installed. Please install first to initialize solver instance."
            raise ImportError(msg)

    def safe_get_solution(
        self, status: Status, func: Callable[[], Solution]
    ) -> Solution:
        """
        Get solution from function call, if status is unknown still try to run it.
        """
        if status.is_ok:
            return func()
        elif status.status == SolverStatus.unknown:
            try:
                logger.warning("Solution status unknown. Trying to parse solution.")
                sol = func()
                status.status = SolverStatus.ok
                logger.warning("Solution parsed successfully.")
                return sol
            except Exception as e:
                logger.error(f"Failed to parse solution: {e}")
        return Solution()

    @abstractmethod
    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
    ) -> Result:
        """
        Abstract method to solve a linear problem from a model.

        Needs to be implemented in the specific solver subclass. Even if the solver
        does not support solving from a model, this method should be implemented and
        raise a NotImplementedError.
        """
        pass

    @abstractmethod
    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
    ) -> Result:
        """
        Abstract method to solve a linear problem from a problem file.

        Needs to be implemented in the specific solver subclass. Even if the solver
        does not support solving from a file, this method should be implemented and
        raise a NotImplementedError.
        """
        pass

    def solve_problem(
        self,
        model: Model | None = None,
        problem_fn: Path | None = None,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: EnvType | None = None,
        explicit_coordinate_names: bool = False,
    ) -> Result:
        """
        Solve a linear problem either from a model or a problem file.

        Wraps around `self.solve_problem_from_model()` and
        `self.solve_problem_from_file()` and calls the appropriate method
        based on the input arguments (`model` or `problem_fn`).
        """
        if problem_fn is not None and model is not None:
            msg = "Both problem file and model are given. Please specify only one."
            raise ValueError(msg)
        elif model is not None:
            return self.solve_problem_from_model(
                model=model,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                env=env,
                explicit_coordinate_names=explicit_coordinate_names,
            )
        elif problem_fn is not None:
            return self.solve_problem_from_file(
                problem_fn=problem_fn,
                solution_fn=solution_fn,
                log_fn=log_fn,
                warmstart_fn=warmstart_fn,
                basis_fn=basis_fn,
                env=env,
            )
        else:
            msg = "No problem file or model specified."
            raise ValueError(msg)

    @property
    def solver_name(self) -> SolverName:
        return SolverName[self.__class__.__name__]


class CBC(Solver[None]):
    """
    Solver subclass for the CBC solver.

    Attributes
    ----------
    **solver_options
        options for the given solver
    """

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
    ) -> Result:
        msg = "Direct API not implemented for CBC"
        raise NotImplementedError(msg)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file using the CBC solver.

        The function reads the linear problem file and passes it to the solver.
        If the solution is successful it returns variable solutions
        and constraint dual values.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path
            Path to the solution file. This is necessary for solving with CBC.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver

        Returns
        -------
        Result
        """
        sense = read_sense_from_problem_file(problem_fn)
        io_api = read_io_api_from_problem_file(problem_fn)

        if solution_fn is None:
            msg = "No solution file specified. For solving with CBC this is necessary."
            raise ValueError(msg)

        if io_api == "mps" and sense == "max" and _new_highspy_mps_layout:
            msg = (
                "CBC does not support maximization in MPS format highspy versions "
                " >=1.7.1"
            )
            raise ValueError(msg)

        # printingOptions is about what goes in solution file
        command = f"cbc -printingOptions all -import {problem_fn} "

        if warmstart_fn:
            command += f"-basisI {warmstart_fn} "

        if self.solver_options:
            command += (
                " ".join(
                    "-" + " ".join([k, str(v)]) for k, v in self.solver_options.items()
                )
                + " "
            )
        command += f"-solve -solu {solution_fn} "

        if basis_fn:
            command += f"-basisO {basis_fn} "

        Path(solution_fn).parent.mkdir(exist_ok=True)

        command = command.strip()

        if log_fn is None:
            p = sub.Popen(command.split(" "), stdout=sub.PIPE, stderr=sub.PIPE)

            if p.stdout is None:
                msg = (
                    f"Command `{command}` did not run successfully. Check if cbc is "
                    " installed and in PATH."
                )
                raise ValueError(msg)

            output = ""
            for line in iter(p.stdout.readline, b""):
                output += line.decode()
            logger.info(output)
            p.stdout.close()
            p.wait()
        else:
            with open(log_fn, "w") as log_f:
                p = sub.Popen(command.split(" "), stdout=log_f, stderr=log_f)
                p.wait()

        with open(solution_fn) as f:
            first_line = f.readline()

        if first_line.startswith("Optimal "):
            status = Status.from_termination_condition("optimal")
        elif "Infeasible" in first_line:
            status = Status.from_termination_condition("infeasible")
        else:
            status = Status(SolverStatus.warning, TerminationCondition.unknown)
        status.legacy_status = first_line

        # Use HiGHS to parse the problem file and find the set of variable names, needed to parse solution
        if "highs" not in available_solvers:
            raise ModuleNotFoundError(
                f"highspy is not installed. Please install it to use {self.solver_name.name} solver."
            )
        h = highspy.Highs()
        h.silent()
        h.readModel(path_to_string(problem_fn))
        variables = {v.name for v in h.getVariables()}

        def get_solver_solution() -> Solution:
            m = re.match(r"Optimal.* - objective value (\d+\.?\d*)$", first_line)
            if m and len(m.groups()) == 1:
                objective = float(m.group(1))
            else:
                objective = np.nan

            with open(solution_fn, "rb") as f:
                trimmed_sol_fn = re.sub(rb"\*\*\s+", b"", f.read())

            df = pd.read_csv(
                io.BytesIO(trimmed_sol_fn),
                header=None,
                skiprows=[0],
                sep=r"\s+",
                usecols=[1, 2, 3],
                index_col=0,
            )
            variables_b = df.index.isin(variables)

            sol = df[variables_b][2]
            dual = df[~variables_b][3]

            return Solution(sol, dual, objective)

        solution = self.safe_get_solution(status=status, func=get_solver_solution)
        solution = maybe_adjust_objective_sign(solution, io_api, sense)

        # Parse the output and get duality gap and solver runtime
        mip_gap, runtime = None, None
        if log_fn is not None:
            with open(log_fn) as log_f:
                output = "".join(log_f.readlines())
        m = re.search(r"\nGap: +(\d+\.?\d*)\n", output)
        if m and len(m.groups()) == 1:
            mip_gap = float(m.group(1))
        m = re.search(r"\nTime \(Wallclock seconds\): +(\d+\.?\d*)\n", output)
        if m and len(m.groups()) == 1:
            runtime = float(m.group(1))
        CbcModel = namedtuple("CbcModel", ["mip_gap", "runtime"])

        return Result(status, solution, CbcModel(mip_gap, runtime))


class HighsJS(Solver[None]):
    """
    Solver subclass for the HiGHS-JS solver (WebAssembly version for Pyodide).

    This solver is designed for use in browser environments via Pyodide and requires
    the highs-js library to be loaded and exposed via the `js_highs_solve` JavaScript function.

    For more information, see:
    - https://lovasoa.github.io/highs-js/
    - PYODIDE_USAGE.md in the repository

    Attributes
    ----------
    **solver_options
        Options for the solver (currently not fully supported)
    """

    def __init__(
        self,
        **solver_options: Any,
    ) -> None:
        super().__init__(**solver_options)

    def solve_problem_from_model(
        self,
        model: Model,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
        explicit_coordinate_names: bool = False,
    ) -> Result:
        """
        Solve a linear problem directly from a linopy model using HiGHS-JS.

        This method extracts the optimization problem from the linopy model,
        converts it to HiGHS JSON format, calls the JavaScript HiGHS solver,
        and parses the result back into linopy format.

        Parameters
        ----------
        model : linopy.Model
            Linopy model for the problem.
        solution_fn : Path, optional
            Path to the solution file (not supported for HiGHS-JS).
        log_fn : Path, optional
            Path to the log file (not supported for HiGHS-JS).
        warmstart_fn : Path, optional
            Path to the warmstart file (not supported for HiGHS-JS).
        basis_fn : Path, optional
            Path to the basis file (not supported for HiGHS-JS).
        env : None, optional
            Environment for the solver.
        explicit_coordinate_names : bool, optional
            Transfer variable and constraint names to the solver (default: False).

        Returns
        -------
        Result
            Optimization result with status, solution, and solver info.
        """
        if solution_fn or log_fn or warmstart_fn or basis_fn:
            logger.warning(
                "HiGHS-JS solver does not support solution_fn, log_fn, warmstart_fn, or basis_fn. "
                "These arguments will be ignored."
            )

        # Call the bridge function
        try:
            status, condition = solve_with_highs_js(model, **self.solver_options)
        except ImportError as e:
            msg = f"HiGHS-JS solver requires Pyodide environment: {e}"
            logger.error(msg)
            status_obj = Status(SolverStatus.error, TerminationCondition.internal_solver_error)
            status_obj.legacy_status = str(e)
            return Result(status_obj, Solution(), None)
        except Exception as e:
            msg = f"HiGHS-JS solver failed: {e}"
            logger.error(msg)
            status_obj = Status(SolverStatus.error, TerminationCondition.internal_solver_error)
            status_obj.legacy_status = str(e)
            return Result(status_obj, Solution(), None)

        # Convert status string to Status object
        status_obj = Status.from_termination_condition(condition)
        status_obj.legacy_status = condition

        # Extract solution from model (solve_with_highs_js already populated it)
        solution = Solution(
            solution=model.solution if hasattr(model, 'solution') else pd.Series(),
            dual=model.dual if hasattr(model, 'dual') else pd.Series(),
            objective=model.objective_value if hasattr(model, 'objective_value') else np.nan,
        )

        return Result(status_obj, solution, None)

    def solve_problem_from_file(
        self,
        problem_fn: Path,
        solution_fn: Path | None = None,
        log_fn: Path | None = None,
        warmstart_fn: Path | None = None,
        basis_fn: Path | None = None,
        env: None = None,
    ) -> Result:
        """
        Solve a linear problem from a problem file.

        Note: File-based solving is not supported for HiGHS-JS. This method
        raises NotImplementedError.

        Parameters
        ----------
        problem_fn : Path
            Path to the problem file.
        solution_fn : Path, optional
            Path to the solution file.
        log_fn : Path, optional
            Path to the log file.
        warmstart_fn : Path, optional
            Path to the warmstart file.
        basis_fn : Path, optional
            Path to the basis file.
        env : None, optional
            Environment for the solver.

        Returns
        -------
        Result

        Raises
        ------
        NotImplementedError
            File-based solving is not supported for HiGHS-JS.
        """
        msg = (
            "File-based solving is not supported for HiGHS-JS. "
            "Use solve_problem_from_model() instead, or use the direct API with model.solve()."
        )
        raise NotImplementedError(msg)

