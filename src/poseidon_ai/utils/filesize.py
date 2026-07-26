"""Utilities for formatting file sizes."""


def format_file_size(size_bytes: int) -> str:
    """Return a human-readable file size."""

    units = ("Bytes", "KB", "MB", "GB", "TB", "PB")
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            if unit == "Bytes":
                return f"{int(size)} {unit}"

            return f"{size:.2f} {unit}"

        size /= 1024

    raise RuntimeError("Unable to format file size.")