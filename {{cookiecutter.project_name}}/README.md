# {{cookiecutter.project_name}}

{{cookiecutter.project_description}}

## Project Structure

- It uses [uv](https://github.com/astral-sh/uv) for python dependency operations and virtual environment management.
- It uses `src` layout, which is the recommended layout for python projects to avoid common [pitfalls](https://blog.ionelmc.ro/2014/05/25/python-packaging/#the-structure).
- It uses `project.toml` instead of `setup.py` and `setup.cfg`. The reasoning is following:
    - As [official setuptools guide](https://github.com/pypa/setuptools/blob/main/docs/userguide/quickstart.rst) says, " configuring new projects via setup.py is discouraged"
    - One of the biggest problems with setuptools is that the use of an executable file (i.e. the setup.py) cannot be executed without knowing its dependencies. And there is really no way to know what these dependencies are unless you actually execute the file that contains the information related to package dependencies.
    - The pyproject.toml file is supposed to solve the build-tool dependency chicken and egg problem since pip itself can read pyproject.yoml along with the version of setuptools or wheel the project requires.
    - The pyproject.toml file was introduced in PEP-518 (2016) as a way of separating configuration of the build system from a specific, optional library (setuptools) and also enabling setuptools to install itself without already being installed. Subsequently PEP-621 (2020) introduces the idea that the pyproject.toml file be used for wider project configuration and PEP-660 (2021) proposes finally doing away with the need for setup.py for editable installation using pip.


## Install

### Default installation

- Install uv - System wide

```bash
make -s install-uv
source ~/.bashrc
```

- Install the project dependencies

```bash
make -s install
```

- After running above command, the project installed in editable mode with all development and test dependencies installed.

### Docker

```bash
# Development build
docker build --tag {{cookiecutter.project_name}} --file docker/Dockerfile --target development .

# Production build
docker build --tag {{cookiecutter.project_name}} --file docker/Dockerfile --target production .
```

- To run command inside the container:

```bash
docker run -it {{cookiecutter.project_name}}:latest bash

# Temporary container
docker run --rm -it {{cookiecutter.project_name}}:latest bash
```

- Development inside the container:
- For more information [watch-versus-bind-mounts](https://docs.docker.com/compose/file-watch/#compose-watch-versus-bind-mounts)

```bash
docker compose up -d {{cookiecutter.project_name}}-dev --watch
```

- Run production image:

```bash
docker compose up -d {{cookiecutter.project_name}}-prod
```

## IDE Setings

### Pycharm

- Line-length: `Editor -> Code Style -> Hard wrap at 88`

#### Inspections

Settings -> Editor -> Inspections -> Python

Enable all except:

- Accessing a protected member of a class or a module
- Assignment can be replaced with augmented assignments
- Classic style class usage
- Incorrect BDD Behave-specific definitions
- No encoding specified for file
- The function argument is equal to the default parameter
- Type checker compatible with Pydantic
- For "PEP 8 coding style violation":
  Ignore = E266, E501
- For "PEP 8 naming convetion violation":
  Ignore = N803

#### Plugins

- Ruff
- Pydantic

### Vscode

- All recommended settings and extensions can be found in `.vscode` directory.

## Useful Makefile commands

```bash
# All available commands
make
make help

# Run all tests
make -s test

# Run specific tests
make -s test-one TEST_MARKER=<TEST_MARKER>

# Remove unnecessary files such as build,test, cache
make -s clean

# Run all pre-commit hooks
make -s pre-commit

# Lint the project
make -s lint

# Profile a file
make -s profile PROFILE_FILE_PATH=<PATH_TO_FILE>
```
