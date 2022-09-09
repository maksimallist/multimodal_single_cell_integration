import setuptools


setuptools.setup(
    name="singlecell",
    version="0.1.0",
    author="Petrov Maksim Andreevich",
    author_email="maksimallist@gmail.com",
    description="An open source library for building and training neural net for single cell task.",
    long_description="-",
    long_description_content_type="text/markdown",
    url="https://github.com/maksimallist/ca_embryogenesis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    packages=['src'],
    python_requires='>=3.8',
)
