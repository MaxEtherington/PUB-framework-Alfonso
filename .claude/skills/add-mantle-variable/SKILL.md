---
name: add-mantle-variable
description: Registers a new base or derived variable in `lib/mantle_variables.py` on the `MantleVariableRegistry` singleton (`variables`). Use when user says 'add mantle variable', 'register a derived variable', 'compute from mantle data', 'add a new mantle feature', or directly modifies `lib/mantle_variables.py`. Handles base variables loaded directly from netCDF and derived variables computed from registered bases (e.g. contour depths, velocity composites, rolling-mean temperature deviations). Do NOT use for adding features to `lib/grid_features.py` — that uses a separate registry.
paths:
  - lib/mantle_variables.py
---
# Add Mantle Variable

## Critical

- **Never edit notebook files directly.** Edit `nb_scripts/*.py` — the `PostToolUse` hook in `.claude/settings.json` automatically runs `jupytext --sync` after each edit.
- All variables live in `lib/mantle_variables.py`. The module-level singleton `variables = MantleVariableRegistry()` must be the target of every registration call.
- Every `@variables.register` function **must** call `.rename("VariableName")` on the returned `xr.DataArray` and set the returned array's `.attrs` with at least `"long_name"` (and `"units"` where applicable).
- Use `variables.get("ExistingVar", ds)` (not `ds["ExistingVar"]`) to retrieve any previously registered variable inside a derived function — this triggers caching and derived-variable computation.
- Loop-based registrations must capture loop variables in the closure: define the function with default-argument binding (e.g. `def _(ds, t=t, b=b):`) to avoid the classic late-binding bug.

## Instructions

### Step 1 — Identify the variable type

Determine which of three types applies:

| Type | When to use | Registration method |
|------|-------------|--------------------|
| **Base** | Variable exists as a named field in the netCDF dataset | `variables.register_base("Name")` |
| **Derived (decorator)** | Computed from one or more existing variables | `@variables.register("Name")` |
| **Derived (loop)** | Multiple related variables sharing a template | `@variables.register(computed_name)` inside a `for` loop |

Verify the netCDF field name (for base) or that all dependency variables are already in `variables.available` (for derived) before proceeding.

### Step 2 — Add a base variable (if applicable)

Append the name to `BASE_MANTLE_VARS` at the top of `lib/mantle_variables.py`, then call `register_base` (already called on that list at module level):

```python
# In lib/mantle_variables.py — top of file
BASE_MANTLE_VARS: list[str] = [
    ...
    "MyNewField_CG",   # add here
]
# register_base(*BASE_MANTLE_VARS) is called automatically below
```

Verify: `"MyNewField_CG" in variables.available` resolves `True` after import.

### Step 3 — Add a derived variable via decorator (if applicable)

Add under the `# Derived variables` section in `lib/mantle_variables.py`:

```python
@variables.register("MyVariable")
def _my_variable(ds: xr.Dataset) -> xr.DataArray:
    base = variables.get("SomeDependency", ds)      # always use variables.get
    result = base.some_operation()
    result.attrs.update({"long_name": "description of variable", "units": "km"})
    return result.rename("MyVariable")             # name must match register arg
```

For derived variables that use `pint` units (e.g. velocity composites), follow the `Speed` / `Tangential_Speed` pattern:

```python
@variables.register("MySpeed")
def _my_speed(ds: xr.Dataset) -> xr.DataArray:
    ds = ds.pint.quantify()
    ve = ds["East_Velocity"]
    vn = ds["North_Velocity"]
    result = np.sqrt(ve**2 + vn**2)
    result.attrs.update({"long_name": "my speed"})
    result = result.pint.dequantify()
    return result.rename("MySpeed")
```

For contour-depth variables, call `_calculate_contour_depth` directly:

```python
@variables.register("MyIsotherm_Depth")
def _my_isotherm_depth(ds: xr.Dataset) -> xr.DataArray:
    result = _calculate_contour_depth(ds, "FullTemperature_CG", target_contour=1500)
    result.attrs.update({"long_name": "depth to 1500K isotherm", "units": "km"})
    return result.rename("MyIsotherm_Depth")
```

Verify: `"MyVariable" in variables.available` and `variables.get("MyVariable", ds)` returns an `xr.DataArray` with correct `.name`.

### Step 4 — Add loop-registered variables (if applicable)

Place under the `# Transformed variables` section. **Bind loop variables as defaults** to avoid closure capture bugs:

```python
for depth_min, depth_max in [(0, 400), (100, 400), (200, 600)]:
    av_name = f"MyVar_Avg_{depth_min}-{depth_max}km"
    @variables.register(av_name)
    def _(ds: xr.Dataset, t=depth_min, b=depth_max) -> xr.DataArray:
        return _average_over_depth(
            ds=ds,
            var_name="Temperature_Deviation_CG",
            depth_range=(t, b),
        )
```

Verify: all expected names appear in `variables.available` after import.

### Step 5 — Register a new transform (if applicable)

Only needed when introducing a reusable computation pattern. Add under `# Transforms`:

```python
@variables.register_transform("my_transform")
def _my_transform(
    ds: xr.Dataset,
    var_name: str,
    my_param: float,
) -> xr.DataArray:
    result = variables.get(var_name, ds)
    if "depth" not in result.dims:
        raise ValueError(f"'{var_name}' has no 'depth' dimension.")
    out = result.some_operation(my_param)
    return out.rename(f"{var_name}_my_transform_{my_param}")
```

Then use it via `register_derived`:

```python
variables.register_derived("MyDerivedVar", "my_transform", var_name="Temperature_CG", my_param=42.0)
```

Verify: `"my_transform" in variables._transforms` and `"MyDerivedVar" in variables.available`.

### Step 6 — Sync notebooks if a variable is referenced there

If a script in `nb_scripts/*.py` references the new variable, edit only the script. The `PostToolUse` hook automatically runs `jupytext --sync` after the edit — no manual sync needed.

### Step 7 — Lint

```bash
ruff check lib/mantle_variables.py
```

## Examples

**User says:** "Add a variable for the depth-averaged viscosity between 100 and 660 km."

**Actions taken:**

1. `Viscosity_CG` is already in `BASE_MANTLE_VARS` — no base registration needed.
2. Add under `# Transformed variables`:

```python
@variables.register("Viscosity_Avg_100-660km")
def _(ds: xr.Dataset) -> xr.DataArray:
    return _average_over_depth(
        ds=ds,
        var_name="Viscosity_CG",
        depth_range=(100, 660),
    )
```

3. `ruff check lib/mantle_variables.py` — clean.

**Result:** `variables.get("Viscosity_Avg_100-660km", ds)` returns a `(time, lat, lon)` `DataArray` named `Viscosity_CG_avg_100-660km`.

---

**User says:** "Register a derived variable for the depth difference between the 1500K isotherm and the LAB."

**Actions taken:**

1. Both `LAB_Depth` and `1000K_Isotherm_Depth` exist — follow the `Sublithospheric_Cold_Anomaly_Thickness` pattern.
2. Add under `# Derived variables`:

```python
@variables.register("LAB_Isotherm_Separation")
def _lab_isotherm_separation(ds: xr.Dataset) -> xr.DataArray:
    result = variables.get("LAB_Depth", ds) - _calculate_contour_depth(ds, "FullTemperature_CG", target_contour=1500)
    result.attrs.update({"long_name": "depth separation between LAB and 1500K isotherm", "units": "km"})
    return result.rename("LAB_Isotherm_Separation")
```

**Result:** Variable is registered, computes without error, and is available via `variables.get`.

## Common Issues

**`KeyError: "Unknown variable: 'MyVar'. Available: [...]`**
The variable was not registered before being retrieved. Check that the `@variables.register("MyVar")` decorator is present *above* the function definition and that there are no import errors above it in the file that silently abort registration.

**DataArray name is `None` or wrong after `variables.get`**
The function returned without calling `.rename("MyVar")`. Every registered function must end with `return result.rename("MyVar")` where the name exactly matches the string passed to `@variables.register`.

**Loop variables all produce the same result (last iteration)**
Closure capture bug. Fix by binding loop variables as default arguments: `def _(ds, t=t, b=b):` instead of `def _(ds):`.

**`AssertionError: Depth must be monotonically decreasing`** in `_calculate_contour_depth`
The dataset depth coordinate is not in descending order. Sort the dataset before passing: `ds = ds.sortby("depth", ascending=False)`.

**`KeyError: "Unknown transform: 'my_transform'"` in `register_derived`**
`register_derived` is called before the `@variables.register_transform("my_transform")` decorator runs. Move the transform definition above the `register_derived` call.

**`ValueError: Data array 'X' has dimension 'depth' but no corresponding coordinates were provided`** during sampling
The derived variable retained a `depth` dimension. If your variable should be depth-reduced (e.g. an average), ensure the computation includes `.mean(dim="depth")` or `.max(dim="depth")` before returning.
