# `HyperAssembly`

## Multi-agent modeling with large language models

**`HyperAssembly`** is a large language model framework designed for multi-agent modeling.

**Multi-agent modeling** is the process of using the actions of various agents in order to
produce statistical models. In our case, these models are created by having agents query and process text
from a robust data corpus, and then "reason" with the information extracted from that text to
produce a data point.

Given the stochastic nature of large language models, each data point may vary - however if a large
enough sample of agents are used, the models that can be produced can be effective for comparative analysis.

The abstraction in this repository that is capable of producing a multi-agent model is referred to as an `Assembly`.
An assembly takes a hypothesis as an argument and returns a probability estimate that hypothesis is correct.

**It is important not to put too much weight into any given probablility estimate, for reasons noted below.
However interesting differences can be observed when varying `Assembly` parameters which can still provide insight
against some independent variable.**

The probability estimates produced are ultimately a form of sentiment analysis against a corpus of text.

With further advances in large language models, and AI broadly, multi-agent modeling may continue to prove a useful
paradigm for classification and regression models, and may become a valuable extra data point for researchers investigating
complex issues with many complex underlying variables.

# Ways to use `HyperAssembly`

## Comparing against a ground truth

## Comparing changes over time

## Measuring bias in different data corpuses

## Setup

This project depends on `poetry` for package management.

Once you have Python 3.11, and Poetry, you can set up the project locally by cloning and then running:

```
poetry install
```

Check out the notebooks for folder for example usage.
