from setuptools import setup, find_packages

setup(
    name="syntheval",  # Replace with your package name
    version="0.1.0",
    description="A brief description of your package",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/my_package",  # Replace with your URL
    packages=find_packages(),
    install_requires=[  # Add dependencies here
        "numpy",
        "pandas"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)