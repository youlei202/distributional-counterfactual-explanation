from setuptools import setup, find_packages

setup(
    name='DistributionalCounterfactualExplaniner',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.,
        # 'numpy>=1.14.0',  
    ],
    author = "Lei You, Lele Cao",
    author_email = "leiyo@dtu.dk, lele.cao@eqtpartners.com",
    description="Package of distributional counterfactual explanation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/youlei202/distributional-counterfactual-explanation",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
    ],
    keywords = "Machine Learning, Data Science, Causal Inference",
    project_urls={
        "Source": 'https://github.com/youlei202/distributional-counterfactual-explanation'
    },
)