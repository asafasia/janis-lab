# Copyright 2014-2021 Keysight Technologies

from setuptools import setup
from Version import __version__


requirements = [
    "numpy",
    "h5py",
    "pyqt5",
    "qtpy",
    "msgpack",
    "sip",
    "future",
    "scipy"
]


setup(
    name='Labber',
    version=__version__,
    packages=['Labber'],
    include_package_data=True,
    url='https://bitbucket.it.keysight.com/projects/LABBER/repos/labber/browse',
    description='Python API for Labber project',
    install_requires = requirements,
    python_requires='>=3.6, <3.10',
    author="Keysight Technologies",
    author_email="labber@keysight.com"
)