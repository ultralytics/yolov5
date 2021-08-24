from pathlib import Path


def prepare_temporary_dir(directory_name: str) -> str:
    directory_path = Path(__file__).parent.joinpath(directory_name)
    directory_path.mkdir(parents=True, exist_ok=True)
    return directory_path.as_posix()
