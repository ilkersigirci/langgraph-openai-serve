# Contributing to `langgraph-openai-serve`

Contributions are welcome, and they are greatly appreciated!
Every little bit helps, and credit will always be given.

You can contribute in many ways:

# Types of Contributions

## Report Bugs

Report bugs at https://github.com/ilkersigirci/langgraph-openai-serve/issues

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

## Fix Bugs

Look through the GitHub issues for bugs.
Anything tagged with "bug" and "help wanted" is open to whoever wants to implement a fix for it.

## Implement Features

Look through the GitHub issues for features.
Anything tagged with "enhancement" and "help wanted" is open to whoever wants to implement it.

## Write Documentation

langgraph-openai-serve could always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at https://github.com/ilkersigirci/langgraph-openai-serve/issues.

If you are proposing a new feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

# Get Started!

Ready to contribute? Here's how to set up `langgraph-openai-serve` for local development.
Please note this documentation assumes you already have `uv` and `Git` installed and ready to go.

1. Fork the `langgraph-openai-serve` repo on GitHub.

2. Clone your fork locally:

```bash
cd <directory_in_which_repo_should_be_created>
git clone git@github.com:YOUR_NAME/langgraph-openai-serve.git
```

3. Now we need to install the environment. Navigate into the directory

```bash
cd langgraph-openai-serve
```

Then, install and activate the environment with:

```bash
uv sync
```

4. Install the Git hooks that run linters and formatters at commit time:

```bash
uv run prek install
```

5. Create a branch for local development:

```bash
git checkout -b name-of-your-bugfix-or-feature
```

Now you can make your changes locally.

6. Don't forget to add test cases for your added functionality to the `tests` directory.

7. When you're done, run the lint and test suites:

```bash
make lint
make test
```

For changes under `demo/`, run its locked standalone checks. If the change also
touches the LGOS API or graph contract, test the demo API with the current
checkout overlay:

```bash
make check-demo
make test-demo-local
```

For documentation changes, also run the strict documentation build:

```bash
make doc-build
```

8. Commit your changes and push your branch to GitHub:

```bash
git add .
git commit -m "Your detailed description of your changes."
git push origin name-of-your-bugfix-or-feature
```

9. Submit a pull request through the GitHub website.

# Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.

2. If the pull request adds functionality, the docs should be updated.
   Put your new functionality into a function with a docstring, and add the feature to the list in `README.md`.
