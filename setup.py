#!/usr/bin/env python
import os
from os.path import exists

from setuptools import setup

import versioneer

NAME = "aemcmc"

# Handle builds of nightly release
if "BUILD_AEMCMC_NIGHTLY" in os.environ:
    nightly = True
    NAME += "-nightly"

    from versioneer import get_versions as original_get_versions

    def get_versions():
        from datetime import datetime, timezone

        suffix = datetime.now(timezone.utc).strftime(r".dev%Y%m%d")
        versions = original_get_versions()
        versions["version"] = versions["version"].split("+")[0] + suffix
        return versions

    versioneer.get_versions = get_versions


setup(
    name=NAME,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Miscellaneous MCMC samplers written in Aesara",
    url="http://github.com/aesara-devs/aemcmc",
    maintainer="Brandon T. Willard",
    maintainer_email="aesara-devs@gmail.com",
    packages=["aemcmc"],
    install_requires=[
        "numpy>=1.18.1",
        "scipy>=1.4.0",
        "aesara>=2.8.3",
        "aeppl>=0.0.40",
        "aehmc>=0.0.10",
        "polyagamma>=1.3.2",
        "cons",
        "logical-unification",
        "etuples",
        "miniKanren",
    ],
    tests_require=["pytest"],
    long_description=open("README.rst").read() if exists("README.rst") else "",
    long_description_content_type="text/x-rst",
    zip_safe=False,
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    keywords=["math", "probability", "numerical", "symbolic", "MCMC"],
)
