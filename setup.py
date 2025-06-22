from setuptools import setup, find_packages

setup(
    name="change-point-detection",
    version="1.0.0",
    description="Change Point Detection package for queueing theory analysis",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "statsmodels"
    ],
    python_requires=">=3.7",
) 