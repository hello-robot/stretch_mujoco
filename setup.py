from setuptools import find_packages, setup

setup(
    name="stretch_mujoco",
    version=None,
    packages=find_packages(),
    install_requires=["mujoco", "numpy<2", "opencv-python"],
    package_data={
        "": ["models/*"],
    },
    author="Hello Robot Inc.",
    author_email="support@hello-robot.com",
    description="A package for interfacing with MuJoCo for Stretch robot simulations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/hello-robot/stretch_mujoco",
    python_requires=">=3.9",
    extras_require={
        "dev": [
            "pre-commit",
            "pytest",
            "flake8",
            "black",
            "mypy",
        ]
    },
)
