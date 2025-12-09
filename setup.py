"""Setup script for LLM Behavior Lab."""

from setuptools import setup, find_packages

setup(
    name="llm-behavior-lab",
    version="0.1.0",
    description="Platform for running repeated experiments on local LLMs to study behavior",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "llama-cpp-python>=0.2.0",
        "pydantic>=2.0.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.11",
    entry_points={
        "console_scripts": [
            "llm-behavior-lab=run_batch:main",
        ],
    },
)
