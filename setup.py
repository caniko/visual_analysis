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

dependency_links = ['https://github.com/CINPLA/visual-stimulation']

setup(
    name="vian",
    install_requires=install_requires,
    dependency_links=dependency_links,
    tests_require=install_requires,
    packages=find_packages(),
    include_package_data=True,
    version='0.1',
)
