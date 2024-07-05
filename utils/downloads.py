# Ultralytics YOLOv5 🚀, AGPL-3.0 license
"""Download utils."""

import logging
import subprocess
import urllib
from pathlib import Path

import requests
import torch


def is_url(url, check=True):
    """
    Determines if a string is a URL and optionally checks its existence online.

    Args:
        url (str): The string that needs to be verified as a URL.
        check (bool, optional): If True, the function will check whether the URL exists online. Defaults to True.

    Returns:
        bool: True if the string is a URL (and exists online if `check` is True), otherwise False.

    Raises:
        AssertionError: If the string does not have the components of a valid URL.
        urllib.request.HTTPError: If an error occurs while trying to access the URL online.

    Notes:
        - The function first checks if the string is properly formatted as a URL.
        - If `check` is set to True, it performs an online check to verify the URL's existence by attempting to open it.

    Example:
        ```python
        url = "https://www.example.com"
        is_valid = is_url(url, check=True)  # Returns True if the URL is valid and accessible
        ```
    """
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (urllib.request.urlopen(url).getcode() == 200) if check else True  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


def gsutil_getsize(url=""):
    """
    Returns the size in bytes of a file at a Google Cloud Storage URL using `gsutil du`.

    Args:
        url (str): The Google Cloud Storage URL of the file to check.

    Returns:
        int: The size of the file in bytes. If the command fails or the output is empty, returns 0.

    Examples:
        ```python
        file_size = gsutil_getsize("gs://bucket_name/file_name")
        print(file_size)  # Outputs file size in bytes or 0 if the file does not exist
        ```

    Notes:
        Ensure that `gsutil` is installed and properly configured in your environment to use this function.
    """
    output = subprocess.check_output(["gsutil", "du", url], shell=True, encoding="utf-8")
    return int(output.split()[0]) if output else 0


def url_getsize(url="https://ultralytics.com/images/bus.jpg"):
    """
    Returns the size in bytes of a downloadable file at a given URL.

    Args:
        url (str): The URL of the file whose size is to be determined. Defaults to
            'https://ultralytics.com/images/bus.jpg'.

    Returns:
        int: The size of the file in bytes, or -1 if the file is not found.

    Raises:
        requests.exceptions.RequestException: If an error occurs while making the HTTP request.

    Examples:
        Example usage to get the size of a file:

        ```python
        size = url_getsize("https://example.com/sample.jpg")
        print(f"File size: {size} bytes")
        ```
    """
    response = requests.head(url, allow_redirects=True)
    return int(response.headers.get("content-length", -1))


def curl_download(url, filename, *, silent: bool = False) -> bool:
    """
    Download a file from a URL to a specified filename using curl.

    Args:
        url (str): The URL from which to download the file.
        filename (str | Path): The local path where the downloaded file should be saved.
        silent (bool, optional): If True, suppress curl's progress output. Defaults to False.

    Returns:
        bool: True if the file was downloaded successfully, False otherwise.

    Examples:
        ```python
        success = curl_download("https://ultralytics.com/images/bus.jpg", "bus.jpg")
        if success:
            print("Download completed successfully.")
        else:
            print("Download failed.")
        ```

    Notes:
        This function uses curl with several options for reliability:
        - `-#` for progress indication.
        - `-L` to follow redirects.
        - `--retry 9` to retry up to 9 times on transient errors.
        - `-C -` to resume incomplete downloads.
    """
    silent_option = "sS" if silent else ""  # silent
    proc = subprocess.run(
        [
            "curl",
            "-#",
            f"-{silent_option}L",
            url,
            "--output",
            filename,
            "--retry",
            "9",
            "-C",
            "-",
        ]
    )
    return proc.returncode == 0


def safe_download(file, url, url2=None, min_bytes=1e0, error_msg=""):
    """
    Downloads a file from a primary URL or an alternative URL to a specified path, ensuring a minimum file size.

    This function attempts to download a file from the given URL to a specified file path. If the download
    fails or the file size is below the specified minimum, it will optionally retry with an alternative URL.
    Incomplete downloads are removed to avoid corruption.

    Args:
        file (str | Path): The target file path where the downloaded file will be saved.
        url (str): The primary URL from which to download the file.
        url2 (str, optional): An alternative URL to use if the primary URL fails. Defaults to None.
        min_bytes (int | float, optional): The minimum acceptable file size in bytes. Defaults to 1e0.
        error_msg (str, optional): Custom error message to log if download fails. Defaults to an empty string.

    Returns:
        None

    Raises:
        AssertionError: If the downloaded file does not exist or its size is smaller than min_bytes after
                        attempting both URLs.

    Notes:
        - Utilizes `torch.hub.download_url_to_file` for downloading and `curl` as a fallback method.
        - Progress logging occurs if the logger level is set accordingly.

    Examples:
        ```python
        safe_download('path/to/save/file.zip', 'https://example.com/file.zip', min_bytes=1024)
        ```

    Related Links:
        - `torch.hub.download_url_to_file` for more information on the primary download method:
          https://pytorch.org/docs/stable/hub.html#torch.hub.download_url_to_file
    """
    from utils.general import LOGGER

    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        LOGGER.info(f"Downloading {url} to {file}...")
        torch.hub.download_url_to_file(url, str(file), progress=LOGGER.level <= logging.INFO)
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        if file.exists():
            file.unlink()  # remove partial downloads
        LOGGER.info(f"ERROR: {e}\nRe-attempting {url2 or url} to {file}...")
        # curl download, retry and resume on fail
        curl_download(url2 or url, file)
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            if file.exists():
                file.unlink()  # remove partial downloads
            LOGGER.info(f"ERROR: {assert_msg}\n{error_msg}")
        LOGGER.info("")


def attempt_download(file, repo="ultralytics/yolov5", release="v7.0"):
    """
    Downloads a file from GitHub release assets or via a direct URL if not found locally, with support for backup
    versions.

    Args:
        file (str | Path): The path to the file to be downloaded or checked locally.
        repo (str, optional): The GitHub repository in the format 'username/repo'. Defaults to 'ultralytics/yolov5'.
        release (str, optional): The specific release tag to download from. Defaults to 'v7.0'.

    Returns:
        Path: The path to the downloaded file.

    Notes:
        - This function first checks if the specified file exists locally. If not found, it attempts to download the file
          either from a given URL or from GitHub release assets.
        - Supports automatic retries and resumes downloads in case of failures.
        - Utilizes the GitHub API to fetch the latest or specified release details and assets.

    Examples:
        ```python
        from ultralytics.utils.general import attempt_download

        file_path = attempt_download('yolov5s.pt')
        print(f'File downloaded to {file_path}')
        ```
    """
    from utils.general import LOGGER

    def github_assets(repository, version="latest"):
        """Fetches GitHub repository release tag and asset names using the GitHub API."""
        if version != "latest":
            version = f"tags/{version}"  # i.e. tags/v7.0
        response = requests.get(f"https://api.github.com/repos/{repository}/releases/{version}").json()  # github api
        return response["tag_name"], [x["name"] for x in response["assets"]]  # tag, assets

    file = Path(str(file).strip().replace("'", ""))
    if not file.exists():
        # URL specified
        name = Path(urllib.parse.unquote(str(file))).name  # decode '%2F' to '/' etc.
        if str(file).startswith(("http:/", "https:/")):  # download
            url = str(file).replace(":/", "://")  # Pathlib turns :// -> :/
            file = name.split("?")[0]  # parse authentication https://url.com/file.txt?auth...
            if Path(file).is_file():
                LOGGER.info(f"Found {url} locally at {file}")  # file already exists
            else:
                safe_download(file=file, url=url, min_bytes=1e5)
            return file

        # GitHub assets
        assets = [f"yolov5{size}{suffix}.pt" for size in "nsmlx" for suffix in ("", "6", "-cls", "-seg")]  # default
        try:
            tag, assets = github_assets(repo, release)
        except Exception:
            try:
                tag, assets = github_assets(repo)  # latest release
            except Exception:
                try:
                    tag = subprocess.check_output("git tag", shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
                except Exception:
                    tag = release

        if name in assets:
            file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)
            safe_download(
                file,
                url=f"https://github.com/{repo}/releases/download/{tag}/{name}",
                min_bytes=1e5,
                error_msg=f"{file} missing, try downloading from https://github.com/{repo}/releases/{tag}",
            )

    return str(file)
