from setuptools import setup

setup(
    name='dragonnfruit',
    version='0.2.0',
    author='Jacob Schreiber',
    author_email='jmschreiber91@gmail.com',
    packages=['dragonnfruit'],
    scripts=[],
    url='https://github.com/jmschrei/dragonnfruit',
    license='LICENSE.txt',
    description='dragonnfruit is a method for analyzing scATAC-seq experiments.',
    install_requires=[
        "numpy >= 1.14.2",
        "scipy >= 1.0.0",
        "pandas >= 1.3.3",
        "pyBigWig >= 0.3.17",
        "torch >= 1.9.0",
    ],
)