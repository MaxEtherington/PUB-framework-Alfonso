from pathlib import Path
import sys

from ruamel.yaml import YAML

from .check_files import check_plate_model

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_CONFIG_PATH = REPO_ROOT / "config" / ".run_config.yml"


def _has_plate_model_files(model_dir: Path) -> bool:
    """Return True if the directory appears to contain a usable plate model."""
    if not model_dir.is_dir():
        return False
    has_rotations = any(model_dir.rglob("*.rot"))
    has_features = any(model_dir.rglob("*.gpml")) or any(model_dir.rglob("*.gpmlz"))
    return has_rotations and has_features


def _resolve_config_paths(config):
    """Resolve relative data_dir and output_dir to absolute paths (repo-root-relative)."""
    all_nb = config.get("all_notebooks", {})
    for key in ("data_dir", "output_dir"):
        val = all_nb.get(key)
        if val and not Path(str(val).strip()).is_absolute():
            all_nb[key] = str(REPO_ROOT / str(val).strip())


def _required_str(mapping, key, context):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required config value: {context}.{key}")
    return value.strip()


def _prepare_config(config_path):
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096

    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {config_path}")

    _resolve_config_paths(config)

    with open(RUN_CONFIG_PATH, "w") as f:
        yaml.dump(config, f)

    all_nb = config.get("all_notebooks", {})
    run_name = all_nb.get("run_name", "baseline")
    output_dir = all_nb.get("output_dir", str(REPO_ROOT / "output"))
    snapshot_dir = Path(output_dir) / run_name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    with open(snapshot_dir / "config_snapshot.yml", "w") as f:
        yaml.dump(config, f)

    print(f"Config:          {config_path}", file=sys.stderr)
    print(f"Run config:      {RUN_CONFIG_PATH}", file=sys.stderr)
    print(f"Config snapshot: {snapshot_dir / 'config_snapshot.yml'}", file=sys.stderr)
    return config


def run_setup(config_path):
    """Prepare directories and plate-model data for a configured run."""
    if config_path is None:
        raise ValueError(
            "--config is required when using --setup.\n"
            "  Example: python run_notebooks.py --setup --config config/notebook_parameters_default.yml"
        )

    config = _prepare_config(config_path)

    all_nb = config.get("all_notebooks", {})
    plate_cfg = all_nb.get("plate_model", {})
    if not isinstance(plate_cfg, dict):
        raise ValueError("Missing required config section: all_notebooks.plate_model")

    data_dir = Path(_required_str(all_nb, "data_dir", "all_notebooks"))
    run_name = _required_str(all_nb, "run_name", "all_notebooks")
    output_dir = Path(_required_str(all_nb, "output_dir", "all_notebooks"))
    plate_model_name = _required_str(plate_cfg, "plate_model_name", "all_notebooks.plate_model")
    use_provided_plate_model = bool(plate_cfg.get("use_provided_plate_model", False))

    recon_data_dir = data_dir / plate_model_name
    plate_model_dir = recon_data_dir / "plate_model"
    run_output_dir = output_dir / run_name

    dirs_to_create = [
        data_dir,
        recon_data_dir,
        run_output_dir,
    ]
    if not use_provided_plate_model:
        dirs_to_create.append(plate_model_dir)

    for directory in dirs_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            raise RuntimeError(f"Failed to create directory: {directory}") from exc

    print(f"Data dir:        {data_dir}", file=sys.stderr)
    print(f"Recon dir:       {recon_data_dir}", file=sys.stderr)
    print(f"Plate model dir: {plate_model_dir}", file=sys.stderr)
    print(f"Output dir:      {run_output_dir}", file=sys.stderr)

    if use_provided_plate_model:
        # If model_dir already exists but is empty/incomplete, force a fresh fetch.
        force_download = not _has_plate_model_files(plate_model_dir)
        check_plate_model(
            str(plate_model_dir),
            verbose=True,
            force=force_download,
        )
        if not _has_plate_model_files(plate_model_dir):
            raise RuntimeError(
                f"Provided plate model download did not produce expected files: {plate_model_dir}"
            )
        print("Plate model:     provided model ready", file=sys.stderr)
    else:
        from .plate_models import get_plate_reconstruction

        # Trigger PMM fetch into the configured model directory.
        get_plate_reconstruction(
            model_name=plate_model_name,
            model_dir=str(plate_model_dir),
        )
        print(f"Plate model:     PMM model '{plate_model_name}' ready", file=sys.stderr)

    print("Setup complete.", file=sys.stderr)
