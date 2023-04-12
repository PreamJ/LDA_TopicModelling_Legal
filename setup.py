from setuptools import setup, find_packages

setup(
    name='python-package',
    version='0.1',
    packages=find_packages(exclude=['app*']),
    description='An lda python package',
    install_requires=['numpy, pandas, gensim, pickle, pythainlp'],
    url='',
    author='Pream J',
    author_email='pream.jun@gmail.com'
)