from setuptools.command.install import install
import os
from huggingface_hub import snapshot_download

class PostInstallCommand(install):
    def run(self):
        print("🚀 Downloading pretrained model ez6moe from Hugging Face...")
        snapshot_download(
            repo_id="coonwg1/ez6moe",
            repo_type="model",
            local_dir="model/ez6moe",
            local_dir_use_symlinks=False
        )
        install.run(self)
