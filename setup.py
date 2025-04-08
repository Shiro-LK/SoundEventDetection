# -*- coding: utf-8 -*-
from setuptools import setup
from io import open

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README


setup(
    name='audio_sed',
    version='0.0.2',    
    description='Implementation of audio SED architecture for tensorflow and pytorch.',
    author='Shiro-LK',
    author_email='shirosaki94@gmail.com',
    license='MIT License',
    packages=['audio_sed', 'audio_sed.pytorch', 'audio_sed.tensorflow'],
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=['numpy', 
                      "pytest", "tensorflow", "torch",
                      ],
    url='https://github.com/Shiro-LK/SoundEventDetection',
    download_url='https://github.com/Shiro-LK/SoundEventDetection.git',
    keywords=["torch", "tensorflow", "audio_sed"],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
