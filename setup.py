from pathlib import Path

from setuptools import find_packages, setup

cur_dir = Path(__file__).absolute().parent
long_description = (cur_dir / "README.md").read_text(encoding="utf-8")
version = (cur_dir / "m2cgen" / "VERSION.txt").read_text(encoding="utf-8").strip()

setup(
    name="m2cgen-refresh",
    version=version,
    url="https://github.com/schen18/m2cgen",
    description=(
        "Refreshed fork of m2cgen: transpile ML models into native code "
        "(Python, Java, JavaScript, C#, PHP, PowerShell, Rust, Visual Basic, SQL)."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(exclude=["tests.*", "tests", "tools"]),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    keywords=("sklearn statsmodels xgboost lightgbm "
              "machine-learning ml regression classification "
              "transpilation code-generation"),
    python_requires=">=3.11",
    install_requires=[
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "m2cgen = m2cgen.cli:main",
        ],
    }
)
