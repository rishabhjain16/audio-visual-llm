from setuptools import setup, find_packages

setup(
    name="avsr_llm",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"":"src"},
    install_requires=[
        line.strip() for line in open("requirements.txt", "r").readlines()
    ],
    author="Rishabh Jain",
    author_email="your.email@example.com",
    description="Audio-Visual Speech Recognition with Large Language Models",
    url="https://github.com/yourusername/AVSR-LLM",
    python_requires=">=3.9",
)