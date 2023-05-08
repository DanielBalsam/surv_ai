# HyperAssembly
## Multi-agent modeling with large language models

**HyperAssembly** is a large language model framework designed for multi-agent modeling.

**Multi-agent modeling** is the process of using the actions of various agents in order to
produce statistical models. In our case, these models are created by having agents query and process text
from a robust data corpus, and then "reason" with the information extracted from that text to
produce a data point.

Given the stochastic nature of large language models, each data point may vary - however if a large
enough sample of these data points is taken, the statistical models that can be produced can have
interesting emergent behaviors. 

The abstraction in this repository that is capable of producing a multi-agent model is referred to as an `Assembly`.

The models produced are ultimately a form of sentiment analysis, and thus the quality of the models is only
as good as the data corpus that is provided. However, as demonstrated in the examples below, trends in `Assembly`
produced models are able to fit to trends observed in other statistical models assessing the same problem area.

With further advances in large language models, and AI broadly, multi-agent modeling may continue to prove a useful
paradigm for classification and regression models, and may become a valuable extra data point for researchers investigating
complex issues with many complex underlying variables.

## Setup

This project depends on `poetry` for package management. 

Once you have Python 3.11, and Poetry, you can set up the project locally by cloning and then running:

```
poetry install
```

Check out the notebooks for folder for example usage.