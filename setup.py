from setuptools import setup, find_packages
from ezscore.hooks import PostInstallCommand

setup(
    name="ezscore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "huggingface_hub>=0.33.0",
        # any other deps
    ],
    cmdclass={
        'install': PostInstallCommand,
    }
)
