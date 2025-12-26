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
IO_APIS = FILE_IO_APIS + ["direct"]1

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
        from linopy.solvers.highs_js import solve_with_highs_js

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

