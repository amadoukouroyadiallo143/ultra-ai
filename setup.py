from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ultra-ai-model",
    version="1.0.0",
    author="Ultra-AI Team",
    author_email="contact@ultra-ai.com",
    description="Revolutionary 390B parameter multimodal AI model with ultra-long context",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ultra-ai/ultra-ai-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.5.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "training": [
            "deepspeed>=0.11.0",
            "accelerate>=0.24.0",
            "wandb>=0.16.0",
            "tensorboard>=2.14.0",
        ],
        "inference": [
            "onnx>=1.14.0",
            "onnxruntime>=1.16.0",
            "triton>=2.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ultra-ai-train=scripts.train:main",
            "ultra-ai-deploy=scripts.deploy:main",
        ],
    },
    include_package_data=True,
    package_data={
        "ultra_ai_model": [
            "src/config/*.yaml",
            "docs/*.md",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/ultra-ai/ultra-ai-model/issues",
        "Source": "https://github.com/ultra-ai/ultra-ai-model",
        "Documentation": "https://ultra-ai-model.readthedocs.io/",
        "Research Paper": "https://arxiv.org/abs/2024.ultra-ai",
    },
)