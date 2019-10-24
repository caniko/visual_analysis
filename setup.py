from setuptools import setup, find_packages
import os


install_requires = [
    'numpy',
    'matplotlib',
    'seaborn',
    'elephant',
    'quantities',
    'neo'
]

dependency_links = [
    'git+https://github.com/CINPLA/visual-stimulation.git',
    'git+https://github.com/caniko2/read_exdir.git'
]

setup(
    name="vian",
    install_requires=install_requires,
    tests_require=install_requires,
    dependency_links=dependency_links,
    packages=find_packages(),
    include_package_data=True,
    version='0.1',
)
