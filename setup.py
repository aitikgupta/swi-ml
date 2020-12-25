import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cu-ml",
    version="0.0.1",
    author="Aitik Gupta",
    author_email="aitikgupta@gmail.com",
    description="A lighweight Machine Learning library which can run on GPU "
    "with switchable backends from 'NumPy' to 'CuPy' "
    "- built from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aitikgupta/cu-ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
