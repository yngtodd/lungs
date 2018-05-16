from setuptools import setup, find_packages


setup(
    name='lungs',
    version="0.0.1",
    packages=find_packages(),
    install_requires=['numpy'],

    # metadata for upload to PyPI
    author="Todd Young",
    author_email="youngmt1@ornl.gov",
    description="Classification and Localization of Thoracic Diseases.",
    license="MIT",
    keywords="Imaging, ML, optimization",
    url="https://github.com/yngtodd/lungs",
)
