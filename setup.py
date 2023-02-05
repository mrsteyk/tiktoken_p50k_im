from setuptools import setup, find_namespace_packages

setup(
    name="tiktoken_p50k_im",
    packages=find_namespace_packages(include=['tiktoken_ext.*']),
    install_requires=["tiktoken"],
    version="2.0.0",
    author="mrsteyk",
)