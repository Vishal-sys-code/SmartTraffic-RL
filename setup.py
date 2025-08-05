import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SmartTraffic-RL",
    version="0.1.0",
    author="Vishal Pandey",
    author_email="pandeyvishal.mlprof@gmail.com",
    description="A Gym-style RL environment for adaptive urban traffic signal control.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/openai/smart-traffic-rl",
    packages=setuptools.find_packages(exclude=["tests", "docs"]),
    install_requires=["gymnasium>=0.26.0", "numpy>=1.19.0"],
    python_requires=">=3.7",
)
