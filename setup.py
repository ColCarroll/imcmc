from codecs import open
from os.path import join, abspath, dirname
from setuptools import setup, find_packages

here = abspath(dirname(__file__))

# Get the long description from the README file
with open(join(here, 'README.md'), encoding='utf-8') as buff:
    long_description = buff.read()

requirements_file = join(here, 'requirements.txt')

with open(requirements_file) as f:
    install_reqs = f.read().splitlines()

setup(
    name='imcmc',
    version='0.1.0',
    description='Turn images into probability distributions, and then sample from them',
    long_description=long_description,
    author='Colin Carroll',
    author_email='colcarroll@gmail.com',
    url='https://github.com/ColCarroll/imcmc',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(exclude=['test']),
    install_requires=install_reqs,
    include_package_data=True,
)
