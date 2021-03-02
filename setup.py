#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cardec",
    version="1.0.2.6",
    author="Justin Lakkis",
    author_email="jlakks@gmail.com",
    description="A deep learning method for joint batch correction, denoting, and clustering of single-cell rna-seq data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jlakkis/CarDEC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.18.1', 'pandas>=1.0.1', 'scipy>=1.4.1', 'tensorflow>= 2.0.1, <=2.3.1', 'scikit-learn>=0.22.2.post1', 'scanpy>=1.5.1', 'louvain>=0.6.1'],
    python_requires='>=3.7',
)