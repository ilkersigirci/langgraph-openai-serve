# LangGraph OpenAI Serve

Welcome to the documentation for `langgraph-openai-serve` - a package that provides an OpenAI-compatible API for LangGraph instances.

## Overview

LangGraph OpenAI Serve allows you to expose your [LangGraph](https://github.com/langchain-ai/langgraph) workflows and agents through an OpenAI-compatible API interface. This enables seamless integration with any client library or tool that works with the OpenAI API, providing a standardized way to interact with your custom LangGraph solutions.

## Features

- Expose your LangGraph instances through an OpenAI-compatible API
- Register multiple graphs and map them to different model names
- Use with any FastAPI application
- Support for both streaming and non-streaming completions
- Docker support for easy deployment

## Table Of Contents

The documentation follows the best practice for project documentation as described by Daniele Procida in the [Diátaxis documentation framework](https://diataxis.fr/) and consists of four separate parts:

1. [Tutorials](tutorials/index.md) - Step-by-step instructions to get you started
2. [How-To Guides](how-to-guides/index.md) - Practical guides for specific tasks
3. [Reference](reference.md) - Technical documentation of the API
4. [Explanation](explanation/index.md) - Conceptual explanations of the architecture

## Installation

```bash
# Using uv
uv add langgraph-openai-serve

# Using pip
pip install langgraph-openai-serve
```

## Quick Links

- [GitHub Repository](https://github.com/ilkersigirci/langgraph-openai-serve)
- [Getting Started](tutorials/getting-started.md)
- [API Reference](reference.md)
- [Docker Deployment](how-to-guides/docker.md)
