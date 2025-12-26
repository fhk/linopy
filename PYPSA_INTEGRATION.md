# PyPSA Integration with linopy HiGHS-JS Solver

## Summary

**Good news: You don't need to change anything in PyPSA!**

Since PyPSA uses linopy for optimization, and we've added HiGHS-JS support directly into linopy, PyPSA will automatically be able to use the HiGHS-JS solver.

## What Changed in linopy

1. **Removed all subprocess calls** - linopy can now be imported in Pyodide
2. **Added HighsJS solver class** - integrates with highs-js via JavaScript
3. **Auto-detection** - Automatically adds "highs-js" to available solvers when in Pyodide

## What You Need to Do

### 1. In Your HTML File

Load highs-js and create the bridge function:

```html
<script src="https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js"></script>
<script src="https://lovasoa.github.io/highs-js/highs.js"></script>

<script>
  let highs = null;

  async function solveWithHiGHS(problem) {
    if (!highs) {
      const highs_loader = require("highs");
      highs = await highs_loader({
        locateFile: (file) => "https://lovasoa.github.io/highs-js/" + file,
      });
    }
    return highs.solve(problem);
  }

  async function main() {
    let pyodide = await loadPyodide();

    // IMPORTANT: Set this BEFORE importing linopy/pypsa
    pyodide.globals.set("js_highs_solve", solveWithHiGHS);

    // Load packages
    await pyodide.loadPackage("micropip");
    const micropip = pyodide.pyimport("micropip");

    // Install your wheels
    await micropip.install("linopy-xxx.whl");
    await micropip.install("pypsa-xxx.whl");

    // Now use PyPSA normally!
    await pyodide.runPythonAsync(`
import pypsa

n = pypsa.Network()
n.add("Bus", "bus1")
n.add("Load", "load1", bus="bus1", p_set=100)
n.add("Generator", "gen1", bus="bus1", p_nom=100, marginal_cost=20)

# This will use highs-js automatically!
n.optimize(solver_name="highs-js")

print("Status:", n.status)
print("Objective:", n.objective)
    `);
  }

  main();
</script>
```

### 2. In PyPSA Code (Optional)

PyPSA's existing code should work as-is. However, you might want to add a check to see if you're in Pyodide and default to highs-js:

```python
# In PyPSA's optimize function, this is already handled by linopy!
# If you want to explicitly use highs-js:

def optimize(self, solver_name=None, **kwargs):
    # Let linopy auto-detect available solvers
    if solver_name is None:
        # linopy will use the first available solver
        # In Pyodide with highs-js setup, this will be "highs-js"
        pass

    # Rest of your existing code...
    return self.model.solve(solver_name=solver_name, **kwargs)
```

## How the Integration Works

### Step-by-Step Flow

1. **User loads HTML with highs-js** → JavaScript library available
2. **User creates `solveWithHiGHS` bridge** → JavaScript function ready
3. **User exposes it via `pyodide.globals.set()`** → Python can access it as `js.js_highs_solve`
4. **User imports PyPSA** → Which imports linopy → Which auto-detects highs-js
5. **User calls `n.optimize()`** → PyPSA calls linopy's solve
6. **linopy dispatches to HighsJS solver** → Extracts model to JSON
7. **HighsJS calls `js.js_highs_solve()`** → JavaScript HiGHS solves it
8. **Result parsed and returned** → Solution stored in PyPSA network

### Under the Hood

```
PyPSA Network
    ↓
calls n.optimize(solver_name="highs-js")
    ↓
PyPSA's linopy Model
    ↓
calls model.solve(solver_name="highs-js")
    ↓
linopy dispatches to HighsJS solver class
    ↓
linopy/solvers/highs_js.py extracts matrices
    ↓
Converts to HiGHS JSON format
    ↓
Calls js.js_highs_solve(problem_dict)
    ↓
JavaScript: highs.solve(problem_dict)
    ↓
Result returned to Python
    ↓
linopy stores solution in model
    ↓
PyPSA has access to solution
```

## Testing in PyPSA

```python
import pypsa
import linopy

# Check available solvers
print("Available solvers:", linopy.solvers.available_solvers)
# Should show: ['highs-js']

# Create simple network
n = pypsa.Network()
n.add("Bus", "bus1")
n.add("Load", "load1", bus="bus1", p_set=100)
n.add("Generator", "gen1", bus="bus1", p_nom=150, marginal_cost=20)
n.add("Generator", "gen2", bus="bus1", p_nom=100, marginal_cost=30)

# Optimize (will use highs-js)
status, condition = n.optimize(solver_name="highs-js")

print(f"Status: {status}, Condition: {condition}")
print(f"Objective: {n.objective}")
print(f"Generator dispatch:\n{n.generators_t.p}")
```

## Differences from Standard PyPSA

### What Works the Same
- ✅ All PyPSA network building
- ✅ All PyPSA components
- ✅ Linear optimal power flow
- ✅ Unit commitment (if linear)
- ✅ All PyPSA results and statistics

### What's Different
- ⚠️ Only linear optimization (no quadratic objectives)
- ⚠️ Solver must be explicitly "highs-js" or left as None to auto-detect
- ⚠️ No warmstart/basis file support
- ⚠️ Can't use CBC, GLPK, or other subprocess-based solvers

## Environment Detection

You can check if you're in a Pyodide environment:

```python
def is_pyodide():
    try:
        import js
        return True
    except ImportError:
        return False

def get_default_solver():
    if is_pyodide():
        return "highs-js"
    else:
        return "highs"  # or "gurobi", etc.
```

## Complete Example

See `PYODIDE_USAGE.md` for a complete working example.

## Building and Deploying

1. **Build linopy wheel** with these changes
2. **Host wheel** on GitHub Pages or CDN
3. **Update PyPSA's HTML examples** to use the new linopy wheel
4. **No changes needed to PyPSA source code!**

## FAQ

**Q: Do I need to modify PyPSA source code?**
A: No! PyPSA uses linopy's solver interface, which now supports highs-js.

**Q: Will this work with my existing PyPSA models?**
A: Yes, as long as they use linear optimization (most PyPSA models do).

**Q: Can I still use regular HiGHS in standard Python?**
A: Yes! The highs-js solver only activates in Pyodide. Regular Python will use highspy as usual.

**Q: What if I forget to set js_highs_solve?**
A: linopy will fail to find any solvers and raise "No solver installed" error.

**Q: Can I use multiple solvers?**
A: In Pyodide, typically only highs-js will be available. But you could potentially have multiple JS solver bridges.
