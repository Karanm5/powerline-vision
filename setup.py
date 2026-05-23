from setuptools import setup, find_packages

setup(
    name="powerline-vision",
    version="1.0.0",
    author="Karan",
    author_email="meena.karan9k@gmail.com",
    description="Automated detection of overhead power line cables from aerial imagery using YOLOv8",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Karanm5/powerline-vision",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy<2",
        "opencv-python-headless>=4.9.0",
        "ultralytics>=8.0.0",
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "python-multipart>=0.0.9",
        "Pillow>=10.0.0",
        "pydantic>=2.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
