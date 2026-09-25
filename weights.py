import gdown
import os

def download_weights():
    if os.path.exists("weights/modeldense1.h5"):
        return
    os.makedirs("weights", exist_ok=True)
    gdown.download(id="1--7p9rRJy7WU4OmomkzM8i0veetZctTT", output="weights/modeldense1.h5", quiet=False)
