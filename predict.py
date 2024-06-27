# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import torch
import subprocess
from PIL import Image
from cog import BasePredictor, Input, Path

MODEL_CACHE = "checkpoints"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["TORCH_HOME"] = MODEL_CACHE
os.environ["HF_DATASETS_CACHE"] = MODEL_CACHE
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = MODEL_CACHE
BASE_URL = f"https://weights.replicate.delivery/default/aura-sr/{MODEL_CACHE}/"

from aura_sr import AuraSR


class ExtendedAuraSR(AuraSR):
    @torch.no_grad()
    def upscale(
        self, image: Image.Image, scale_factor: int, max_batch_size=8
    ) -> Image.Image:
        if scale_factor not in [2, 4, 8, 16, 32]:
            raise ValueError("Scale factor must be one of 2, 4, 8, 16, or 32")

        # Use the existing upscale_4x method
        upscaled_image = self.upscale_4x(image, max_batch_size)

        # Adjust the output based on the scale factor
        if scale_factor == 2:
            upscaled_image = upscaled_image.resize(
                (image.width * 2, image.height * 2), Image.BICUBIC
            )
        elif scale_factor in [8, 16, 32]:
            extra_scale = scale_factor // 4
            upscaled_image = upscaled_image.resize(
                (image.width * scale_factor, image.height * scale_factor), Image.BICUBIC
            )

        return upscaled_image


def download_weights(url: str, dest: str) -> None:
    start = time.time()
    print("[!] Initiating download from URL: ", url)
    print("[~] Destination path: ", dest)
    if ".tar" in dest:
        dest = os.path.dirname(dest)
    command = ["pget", "-vf" + ("x" if ".tar" in url else ""), url, dest]
    try:
        print(f"[~] Running command: {' '.join(command)}")
        subprocess.check_call(command, close_fds=False)
    except subprocess.CalledProcessError as e:
        print(
            f"[ERROR] Failed to download weights. Command '{' '.join(e.cmd)}' returned non-zero exit status {e.returncode}."
        )
        raise
    print("[+] Download completed in: ", time.time() - start, "seconds")


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        if not os.path.exists(MODEL_CACHE):
            os.makedirs(MODEL_CACHE)

        model_files = [
            "models--fal-ai--AuraSR.tar",
        ]
        for model_file in model_files:
            url = BASE_URL + model_file
            filename = url.split("/")[-1]
            dest_path = os.path.join(MODEL_CACHE, filename)
            if not os.path.exists(dest_path.replace(".tar", "")):
                download_weights(url, dest_path)

        # Load the pre-trained model with its original configuration
        self.model = ExtendedAuraSR.from_pretrained()

    @torch.no_grad()
    def predict(
        self,
        image: Path = Input(description="The input image file to be upscaled."),
        scale_factor: int = Input(
            description="The factor by which to upscale the image (2, 4, 8, 16, or 32).",
            choices=[2, 4, 8, 16, 32],
            default=4,
        ),
        max_batch_size: int = Input(
            description="Controls the number of image tiles processed simultaneously. Higher values may increase speed but require more GPU memory. Lower values use less memory but may increase processing time. Default is 1 for broad compatibility. Adjust based on your GPU capabilities for optimal performance.",
            default=1,
            ge=1,
            le=8,
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        # Load and process the input image
        input_image = Image.open(str(image))

        # Perform the upscaling
        upscaled_image = self.model.upscale(
            input_image, scale_factor, max_batch_size=max_batch_size
        )

        # Save the output image
        output_path = Path("/tmp/output.png")
        upscaled_image.save(str(output_path))

        return output_path
