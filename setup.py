from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="sdcpt",
    version="0.0.2",
    author="simudt",
    author_email="dogukanuraztuna@gmail.com",
    description="Spatial distance computations functions provided for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simudt/SDCPT",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
    ],
    keywords="pytorch spatial distance computations",
)
