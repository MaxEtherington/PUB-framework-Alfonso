"""Functions to download data bundles from Zenodo, if required."""
import os
from pathlib import Path
from shutil import unpack_archive
from sys import stderr
from tempfile import TemporaryDirectory
from typing import Optional, Union

import requests
from tqdm import tqdm

ROOT = Path(__file__).parent.absolute()


_DOI_URL = "https://doi.org/10.5281/zenodo.8157690"
_ZENODO_URL = "https://zenodo.org/record/14010839"
_TOPOGRAPHY_URL = "https://www.earthbyte.org/webdav/ftp/earthbyte/Paleotopography/paleotopography-data.tgz"


def check_prepared_data(
    data_dir: Union[os.PathLike, str],
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the prepared training and grid data bundle.

    Parameters
    ----------
    data_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    data_dir : str
        The location of the downloaded data bundle.
    """
    data_dir = Path(data_dir).resolve()

    if force or (not data_dir.is_dir()) or (not any(data_dir.iterdir())):
        try:
            zenodo_url = requests.get(_DOI_URL, timeout=5).url
        except Exception:
            zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/prepared_data.zip"

        if verbose:
            print(
                f"Downloading source data: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=os.path.dirname(data_dir),
            verbose=verbose,
        )
    return data_dir


def check_source_data(
    data_dir: Union[os.PathLike, str],
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the source data bundle.

    Parameters
    ----------
    data_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    data_dir : str
        The location of the downloaded data bundle.
    """
    data_dir = Path(data_dir).resolve()

    if force or (not data_dir.is_dir()) or (not any(data_dir.iterdir())):
        try:
            zenodo_url = requests.get(_DOI_URL, timeout=5).url
        except Exception:
            zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/source_data.zip"

        if verbose:
            print(
                f"Downloading source data: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=os.path.dirname(data_dir),
            verbose=verbose,
        )
    return data_dir

def check_erodep_data(
    data_dir: Union[os.PathLike, str],
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the erodep data bundle.

    Parameters
    ----------
    data_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    data_dir : str
        The location of the downloaded data bundle.
    """
    data_dir = Path(data_dir).resolve()
    erodep_filename = data_dir / "erosion_deposition_files.zip"

    if force or (not erodep_filename.is_file()):
        zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/erosion_deposition_files.zip"

        if verbose:
            print(
                f"Downloading erosion/deposition data: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=data_dir,
            verbose=verbose,
        )
    return data_dir


def check_paleotopography_data(
    data_dir: Union[os.PathLike, str],
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the palaeotopography data bundle.

    Parameters
    ----------
    data_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    data_dir : str
        The location of the downloaded data bundle.
    """
    data_dir = Path(data_dir).resolve()
    palaeotopo_filename = data_dir / "paleotopography-data.tgz"

    if force or (not palaeotopo_filename.is_file()):
        url = _TOPOGRAPHY_URL

        if verbose:
            print(
                f"Downloading palaeotopography data: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=data_dir,
            verbose=verbose,
        )
    return data_dir


def check_plate_model(
    model_dir: Union[os.PathLike, str],
    verbose: bool = False,
    force: bool = False,
) -> str:
    """Download the plate model data bundle.

    Parameters
    ----------
    model_dir : str
        Directory in which to place the data bundle.
    verbose : bool, default: False
        Print log to stderr.
    force : bool, default: False
        Download data bundle even if it is already present.

    Returns
    -------
    model_dir : str
        The location of the downloaded data bundle.
    """
    model_dir = Path(model_dir).resolve()

    if force or (not model_dir.is_dir()) or (not any(model_dir.iterdir())):
        try:
            zenodo_url = requests.get(_DOI_URL, timeout=5).url
        except Exception:
            zenodo_url = _ZENODO_URL
        url = f"{zenodo_url}/files/plate_model.zip"

        if verbose:
            print(
                f"Downloading plate model: {url}",
                file=stderr,
                flush=True,
            )
        _download_extract(
            url=url,
            extract_dir=model_dir,
            verbose=verbose,
        )
    return model_dir

def _download_extract(
    url: str,
    extract_dir: Union[os.PathLike, str],
    archive_filename: Optional[Union[os.PathLike, str]] = None,
    verbose: bool = False,
):
    url_ext = Path(url).suffix
    if url_ext not in [".zip", ".tar", ".gz", ".tgz"]:
        raise ValueError(f"URL must point to a .zip, .tar, .gz, or .tgz archive file (got {url_ext})")
    
    if archive_filename is None:
        tempdir = TemporaryDirectory()
        download_dir = tempdir.name
        basename = "data" + url_ext
        archive_filename = os.path.join(
            download_dir,
            basename,
        )
    else:
        tempdir = None
        download_dir = os.path.dirname(os.path.abspath(archive_filename))
        basename = os.path.basename(archive_filename)
    archive_filename = _fetch_data(
        url=url,
        download_dir=download_dir,
        filename=basename,
        verbose=verbose,
    )
    unpack_archive(archive_filename, extract_dir=extract_dir)
    if tempdir is not None:
        tempdir.cleanup()


def _fetch_data(url, download_dir, filename=None, verbose=False):
    # Don't print progress if in a notebook
    # (can result in hundreds of lines of output)
    verbose = verbose and ("get_ipython" not in dir())

    if filename is None:
        filename = "data.zip"
    filename = os.path.join(download_dir, filename)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(filename, "wb") as f:
            it = response.iter_content(chunk_size=1024)
            if verbose:
                it = tqdm(it, unit="kB")
            for data in it:
                f.write(data)
        return filename


if __name__ == "__main__":
    raise SystemExit("Pass explicit paths when calling check_* functions.")
