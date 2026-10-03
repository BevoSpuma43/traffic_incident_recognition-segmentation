"""Local video selection and persistent storage for frontend uploads."""

import hashlib
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile

VIDEO_EXTENSIONS = (
    "mp4",
    "avi",
    "mov",
    "mkv",
    "webm",
    "m4v",
    "mpeg",
    "mpg",
    "mts",
    "m2ts",
    "ts",
    "wmv",
    "flv",
    "3gp",
    "ogv",
    "mxf",
    "h264",
    "h265",
)


def list_dataset_videos(root):
    root = Path(root)
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower().lstrip(".") in VIDEO_EXTENSIONS
    )


def save_uploaded_video(name, content, directory):
    """Keep originals across reruns and never overwrite different same-name videos."""
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")
    if Path(name).suffix.lower().lstrip(".") not in VIDEO_EXTENSIONS:
        raise ValueError("Seleziona un file video con un'estensione supportata.")
    if not content:
        raise ValueError("Il video caricato è vuoto.")
    if name.split(".")[0].upper() in (
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    ):
        name = "video_" + name
    destination = Path(directory).resolve() / hashlib.sha256(content).hexdigest() / name
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
