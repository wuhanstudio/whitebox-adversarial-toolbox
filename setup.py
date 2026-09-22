import codecs
import os

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r", encoding="utf-8") as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

install_requires = [
    "numpy>=1.18.0,<1.20",
    "tqdm",
    "six",
    "setuptools",
    "torch<2.0",
    "torchvision",
    "opencv-python",
    "scikit-image",
    "tensorflow",
    "keras==2.4.3",
    "h5py==2.10.0",
    "protobuf<3.20",
    "matplotlib",
    "pandas",
    "click",
    "progressbar",
    "loguru",
    "psutil",
    "pycocotools",
    "tabulate",
]

setuptools.setup(
    name='whitebox-adversarial-toolbox',
    version=get_version(os.path.join("what", "__init__.py")),
    author="wuhanstudio",
    author_email="wuhanstudios@gmail.com",
    maintainer="wuhanstudio",
    maintainer_email="wuhanstudios@gmail.com",
    description='White-box Adversarial Toolbox (WHAT) - Python Library for Deep Learning Security',
    url="https://github.com/wuhanstudio/whitebox-adversarial-toolbox",
    license="MIT",
    install_requires=install_requires,
    extras_require = {
        "dev": [
            "pytest",
            "pdoc",
            "build",
        ],
        "test": [
            "pytest",
        ],
        "docs": [
            "pdoc",
        ],
    },
    entry_points={
        'console_scripts': [
            'what=what._main:main',
        ],
    },
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
