"""Packaging for the graphdb engine."""

from setuptools import find_packages, setup

setup(
    name="graphdb",
    version="0.1.0",
    description="A from-scratch in-memory property graph database with "
    "label-scoped vector similarity.",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.8",
    install_requires=[],  # numpy optional; pure-python fallback provided
    extras_require={
        "fast": ["numpy>=1.24"],
        "test": ["pytest>=7.0"],
    },
    py_modules=["cli"],
    entry_points={
        "console_scripts": [
            "graphdb-cli=cli:main",
        ],
    },
)
