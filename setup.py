#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='bert4keras',
    version='0.1.7',
    description='更清爽的bert4keras',
    license='MIT Licence',
    url='https://kexue.fm',
    author='bojone',
    author_email='bojone@spaces.ac.cn',
    install_requires=['keras'],
    packages=find_packages()
)
