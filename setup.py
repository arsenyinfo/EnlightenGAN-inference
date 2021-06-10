import os

from setuptools import find_packages, setup

# Package meta-data.
NAME = "enlighten_inference"
DESCRIPTION = "Python library for EnlightenGAN inference"
URL = "https://github.com/arsenyinfo/EnlightenGAN-inference"
REQUIRES_PYTHON = ">=3.6.0"

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def load_requirements(filename):
    with open(os.path.join(PROJECT_ROOT, filename), "r") as f:
        lineiter = f.read().splitlines()

    return [line for line in lineiter if line and not line.startswith("#")]


setup(
    name=NAME,
    version='0.0.1',
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords=[
        "Machine Learning",
        "Deep Learning",
        "Computer Vision",
        "ONNX"
    ],
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={'enlighten_inference': ['*.onnx']},
    install_requires=load_requirements("requirements.txt"),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
