#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

src_dir = os.path.join(os.getcwd(), 'src')
packages = {"" : "src"}
for package in find_packages("src"):
    packages[package] = "src"

setup(
    packages = packages.keys(),
    package_dir = {"" : "src"},
    name = 'unsegqc',
    version = '1.0.0',
    author = 'Benoît Audelan',
    author_email = 'benoit.audelan@inria.fr',
    description = 'Unsupervised Segmentation Quality Control',
    license = 'Inria',
    )
