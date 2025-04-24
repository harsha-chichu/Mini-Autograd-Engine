import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="autpgrad",
    version="0.1.0",
    author="Harsha Vardhan",
    author_email="harshachinnu129@gmail.com",
    description="A tiny scalar-valued autograd engine with a small PyTorch-like neural network library on top.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/micrograd",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)