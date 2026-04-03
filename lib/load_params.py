from copy import deepcopy
from ruamel.yaml import YAML

from .paths import PathConfigManager

DEFAULT_CONFIG_PATH = PathConfigManager.CONFIG_DIR / "notebook_parameters_default.yml"


def get_params(
    config_path=DEFAULT_CONFIG_PATH,
    notebook=None,
):
    yaml = YAML()
    with open(config_path, "r") as f:
        data = yaml.load(f)

    defaults = data["defaults"]

    # Parameters for all notebooks
    ## Start with defaults
    params = deepcopy(defaults.get("all_notebooks", dict()))
    ## Override with values for all notebooks
    params.update(data.get("all_notebooks", dict()))
    if notebook in {None, "all", "all_notebooks"}:
        return params

    # Parameters for specific notebooks
    notebook = str(notebook).lower()
    if not notebook.startswith("notebook_"):
        notebook = "notebook_" + notebook
    ## Add defaults for specific notebook
    for key, value in defaults.get(notebook, dict()).items():
        if key not in params.keys():
            params[key] = value
    ## Add specified values for notebook
    for key, value in data.get(notebook, dict()).items():
        if (key not in params.keys()) or (value is not None):
            params[key] = value
    return params