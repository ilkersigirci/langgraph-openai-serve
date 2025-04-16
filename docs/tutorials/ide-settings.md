# IDE Setings

## Pycharm

- Line-length: `Editor -> Code Style -> Hard wrap at 88`

### Inspections

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

### Plugins

- Ruff
- Pydantic

## Vscode

- All recommended settings and extensions can be found in `.vscode` directory.
