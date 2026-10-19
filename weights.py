import os
import shutil
from pathlib import Path
from urllib.request import Request, urlopen

BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "modeldense1.h5"
WEIGHTS_URL_ENV = "MODEL_WEIGHTS_URL"
GDRIVE_FILE_ID = "1--7p9rRJy7WU4OmomkzM8i0veetZctTT"


def download_weights(dest=WEIGHTS_PATH, url=None):
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = url or os.environ.get(WEIGHTS_URL_ENV)
    partial = dest.with_name(dest.name + ".part")

    try:
        if url:
            _download_url(url, partial)
        else:
            try:
                import gdown
            except ImportError as exc:
                raise RuntimeError(
                    "gdown is not installed. Install it or set MODEL_WEIGHTS_URL to a "
                    "direct download URL."
                ) from exc

            result = gdown.download(id=GDRIVE_FILE_ID, output=str(partial), quiet=False)
            if result is None:
                raise RuntimeError(
                    "Could not download model weights. Set MODEL_WEIGHTS_URL to a direct "
                    "download URL, or make the Google Drive file publicly accessible."
                )

        if not partial.exists() or partial.stat().st_size == 0:
            raise RuntimeError("Downloaded model weights are empty.")

        partial.replace(dest)
    finally:
        if partial.exists():
            partial.unlink()

    return dest


def _download_url(url, dest):
    request = Request(url, headers={"User-Agent": "oncology-breast-scanner/1.0"})
    with urlopen(request, timeout=120) as response, open(dest, "wb") as output:
        shutil.copyfileobj(response, output)
