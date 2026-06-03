from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Union

import joblib
import pandas as pd

PathLike = Union[str, Path]
Operation = Literal["save", "load"]
FileType = Literal["json", "csv", "figure", "joblib"]


def _ensure_directory(directory: PathLike) -> Path:
	"""Create a directory if needed and return it as a Path object."""
	dir_path = Path(directory).expanduser().resolve()
	dir_path.mkdir(parents=True, exist_ok=True)
	return dir_path


def _build_file_path(directory: PathLike, filename: str) -> Path:
	"""Build an absolute file path for a file inside a given directory."""
	if not filename:
		raise ValueError("filename must be a non-empty string")
	return _ensure_directory(directory) / filename


def file_io(
	operation: Operation,
	file_type: FileType,
	directory: PathLike,
	filename: str,
	data: Any = None,
	**kwargs: Any,
) -> Any:
	"""
	Unified file save/load utility.

	Parameters
	----------
	operation : {"save", "load"}
		Whether to write data to disk or read from disk.
	file_type : {"json", "csv", "figure"}
		The type of file to save or load.
	directory : str or Path
		Target directory that contains the file.
	filename : str
		Name of the file (including extension).
	data : Any, optional
		Payload used for save operations.
	**kwargs : Any
		Additional options passed to serializer/deserializer.
	"""
	valid_operations = {"save", "load"}
	valid_file_types = {"json", "csv", "figure", "joblib"}

	if operation not in valid_operations:
		raise ValueError(f"operation must be one of {sorted(valid_operations)}")
	if file_type not in valid_file_types:
		raise ValueError(f"file_type must be one of {sorted(valid_file_types)}")

	file_path = _build_file_path(directory, filename)

	if operation == "save":
		if file_type == "json":
			indent = kwargs.pop("indent", 2)
			ensure_ascii = kwargs.pop("ensure_ascii", False)
			with file_path.open("w", encoding="utf-8") as file_obj:
				json.dump(data, file_obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
			return file_path

		if file_type == "joblib":
			compress = kwargs.pop("compress", 3)
			joblib.dump(data, file_path, compress=compress)
			return file_path

		if file_type == "csv":
			if not isinstance(data, pd.DataFrame):
				raise TypeError("data must be a pandas DataFrame when saving csv")
			index = kwargs.pop("index", False)
			data.to_csv(file_path, index=index, **kwargs)
			return file_path

		# file_type == "figure"
		dpi = kwargs.pop("dpi", 300)
		bbox_inches = kwargs.pop("bbox_inches", "tight")
		if hasattr(data, "savefig"):
			data.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
			return file_path
		if isinstance(data, (bytes, bytearray)):
			file_path.write_bytes(bytes(data))
			return file_path
		raise TypeError("data must be a matplotlib Figure-like object or bytes when saving figure")

	# operation == "load"
	if file_type == "json":
		with file_path.open("r", encoding="utf-8") as file_obj:
			return json.load(file_obj, **kwargs)

	if file_type == "csv":
		return pd.read_csv(file_path, **kwargs)

	if file_type == "joblib":
		return joblib.load(file_path)

	# file_type == "figure"
	return file_path.read_bytes()


def save_json(
	data: Any,
	directory: PathLike,
	filename: str,
	*,
	indent: int = 2,
) -> Path:
	"""Save a JSON-serializable mapping to a file in the target directory."""
	return file_io("save", "json", directory, filename, data=data, indent=indent, ensure_ascii=False)


def read_json(directory: PathLike, filename: str) -> dict[str, Any]:
	"""Read a JSON file from the target directory and return a dictionary."""
	return file_io("load", "json", directory, filename)


def save_csv(
	dataframe: pd.DataFrame,
	directory: PathLike,
	filename: str,
	*,
	index: bool = False,
) -> Path:
	"""Save a DataFrame as CSV to a file in the target directory."""
	return file_io("save", "csv", directory, filename, data=dataframe, index=index)


def read_csv(directory: PathLike, filename: str, **kwargs: Any) -> pd.DataFrame:
	"""Read a CSV file from the target directory into a DataFrame."""
	return file_io("load", "csv", directory, filename, **kwargs)


def save_figure(
	figure: Any,
	directory: PathLike,
	filename: str,
	*,
	dpi: int = 300,
	bbox_inches: str = "tight",
) -> Path:
	"""Save a matplotlib figure to a file in the target directory."""
	return file_io("save", "figure", directory, filename, data=figure, dpi=dpi, bbox_inches=bbox_inches)


def read_figure(directory: PathLike, filename: str) -> bytes:
	"""Read a saved figure/image file as raw bytes."""
	return file_io("load", "figure", directory, filename)


def save_joblib(data: Any, directory: PathLike, filename: str, *, compress: int = 3) -> Path:
	"""Save a Python object with joblib to a file in the target directory."""
	return file_io("save", "joblib", directory, filename, data=data, compress=compress)


def read_joblib(directory: PathLike, filename: str) -> Any:
	"""Load a Python object with joblib from the target directory."""
	return file_io("load", "joblib", directory, filename)
