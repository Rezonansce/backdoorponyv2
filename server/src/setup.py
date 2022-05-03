import codecs
import os

from setuptools import find_packages, setup

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


setup(
    name="backdoorpony",
    version=get_version("backdoorpony/__init__.py"),
    description="GUI toolkit for testing backdoor attack on neural networks.",
    author="BackdoorPony",
    author_email="backdoor@pony.nl",
    maintainer="BackdoorPony",
    maintainer_email="backdoor@pony.nl",
    url="https://gitlab.ewi.tudelft.nl/cse2000-software-project/2020-2021-q4/cluster-14/aisylab-backdoor-attacks-on-neural-networks/backdoor-pony",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
)
