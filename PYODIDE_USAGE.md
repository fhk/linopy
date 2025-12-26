# Using linopy with HiGHS-JS in Pyodide

This document explains how to use linopy with the HiGHS-JS (WebAssembly) solver in browser environments via Pyodide.

## Changes Made

### 1. Removed subprocess dependencies
- All `subprocess` calls have been removed from `linopy/solvers.py`
- CBC and GLPK solvers now raise `NotImplementedError` as they require subprocess
- The module can now be imported in Pyodide without errors

### 2. Added HiGHS-JS solver
- New solver class `HighsJS` in `linopy/solvers.py`
- New bridge module `linopy/solvers/highs_js.py` that handles the JavaScript interop
- Automatically detected when running in Pyodide with `js_highs_solve` available
- Added to `SolverName` enum as `HighsJS = "highs-js"`

## Usage in Pyodide/Browser

### HTML Setup

```html
<!DOCTYPE html>
<html>
  <head>
    <title>linopy with HiGHS-JS</title>
    <script src="https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"></script>
    <script src="https://lovasoa.github.io/highs-js/highs.js"></script>
  </head>
  <body>
    <script>
      // Initialize HiGHS solver
      let highs = null;

      // Bridge function: solve LP/MIP problem using HiGHS
      async function solveWithHiGHS(problem) {
        if (!highs) {
          const highs_loader = require("highs");
          highs = await highs_loader({
            locateFile: (file) => "https://lovasoa.github.io/highs-js/" + file,
          });
        }

        // Solve the problem (problem is a dict with cols, rows, sense, etc.)
        const result = highs.solve(problem);
        return result;
      }

      // Initialize Pyodide and linopy
      async function main() {
        let pyodide = await loadPyodide();

        // Make the solver available to Python
        pyodide.globals.set("js_highs_solve", solveWithHiGHS);

        // Install linopy
        await pyodide.loadPackage("micropip");
        const micropip = pyodide.pyimport("micropip");
        await micropip.install("your-linopy-wheel.whl");

        // Now you can use linopy with highs-js solver
        await pyodide.runPythonAsync(`
import linopy

# Create a model
m = linopy.Model()

# Add variables
x = m.add_variables(lower=0, name="x")
y = m.add_variables(lower=0, name="y")

# Add constraints
m.add_constraints(x + y >= 1, name="c1")

# Set objective
m.add_objective(2 * x + 3 * y)

# Solve with HiGHS-JS
m.solve(solver_name="highs-js")

print("Status:", m.status)
print("Objective:", m.objective.value)
print("Solution x:", x.solution)
print("Solution y:", y.solution)
        `);
      }

      main();
    </script>
  </body>
</html>
```

### PyPSA Integration

For PyPSA, you don't need to change anything in PyPSA code! Since PyPSA uses linopy internally, once you:

1. Load highs-js in your HTML
2. Expose `js_highs_solve` to Python via `pyodide.globals.set()`
3. Install the updated linopy wheel

Then you can simply use:

```python
import pypsa

n = pypsa.Network()
# ... add components ...

# This will automatically use highs-js if it's the only/default solver available
n.optimize(solver_name="highs-js")
```

## How It Works

### Solver Detection

When linopy is imported in Pyodide, it checks if `js.js_highs_solve` is available:

```python
try:
    import js
    if hasattr(js, 'js_highs_solve'):
        available_solvers.append("highs-js")
except (ImportError, AttributeError):
    pass
```

### Solver Bridge

The `linopy/solvers/highs_js.py` module:

1. Extracts the constraint matrix, bounds, and objective from the linopy model
2. Converts it to HiGHS JSON format:
   ```python
   {
     "sense": "minimize",  # or "maximize"
     "offset": 0.0,
     "cols": [{"name": "x0", "obj": 2.0, "lb": 0.0}, ...],
     "rows": [{"coeffs": [{"col": 0, "val": 1.0}, ...], "lb": 1.0}, ...]
   }
   ```
3. Calls the JavaScript `js_highs_solve(problem)` function
4. Parses the result and stores it back in the linopy model

### Supported Features

- ✅ Linear programming (LP)
- ✅ Mixed-integer linear programming (MILP)
- ✅ Variable bounds
- ✅ Equality and inequality constraints
- ✅ Minimize and maximize objectives
- ✅ Dual values
- ❌ Quadratic programming (QP) - HiGHS-JS may not support this
- ❌ Warmstart/basis files
- ❌ File-based solving (use direct API only)

## Checking Available Solvers

```python
import linopy.solvers

print("Available solvers:", linopy.solvers.available_solvers)
# In Pyodide with highs-js: ['highs-js']
# In standard Python with highspy: ['highs', ...]
```

## Troubleshooting

### "No solver installed" error

Make sure `js_highs_solve` is exposed before importing linopy:

```javascript
// Do this BEFORE loading/importing linopy
pyodide.globals.set("js_highs_solve", solveWithHiGHS);
```

### ImportError about js module

This means you're not running in Pyodide. The highs-js solver only works in browser environments.

### "HighsJS solver requires Pyodide" error

Make sure:
1. You've loaded highs-js: `<script src="https://lovasoa.github.io/highs-js/highs.js"></script>`
2. You've created and exposed the `js_highs_solve` function
3. The function is exposed before importing linopy

## Example JavaScript Bridge Function

```javascript
async function solveWithHiGHS(problem) {
  // Initialize HiGHS if needed
  if (!window.highs) {
    const highs_loader = require("highs");
    window.highs = await highs_loader({
      locateFile: (file) => "https://lovasoa.github.io/highs-js/" + file,
    });
  }

  // Solve the problem
  const result = window.highs.solve(problem);

  // Return result in expected format
  return {
    Status: result.Status,
    ObjectiveValue: result.ObjectiveValue,
    Columns: result.Columns,
    Rows: result.Rows
  };
}
```
