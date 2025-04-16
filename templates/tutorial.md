#
-   [Link](https://github.com/NilsJPWerner/autoDocstring/tree/f7bc9f427d5ebcd87e6f5839077a87ecd1cbb404/src/docstring/templates)

# Variables

```mustache
{{name}}                        - name of the function
{{summaryPlaceholder}}          - _summary_ placeholder
{{extendedSummaryPlaceholder}}  - [extended_summary] placeholder
```

# Sections

```mustache
{{#args}}                       - iterate over function arguments
    {{var}}                     - variable name
    {{typePlaceholder}}         - _type_ or guessed type  placeholder
    {{descriptionPlaceholder}}  - _description_ placeholder
{{/args}}

{{#kwargs}}                     - iterate over function kwargs
    {{var}}                     - variable name
    {{typePlaceholder}}         - _type_ or guessed type placeholder
    {{&default}}                - default value (& unescapes the variable)
    {{descriptionPlaceholder}}  - _description_ placeholder
{{/kwargs}}

{{#exceptions}}                 - iterate over exceptions
    {{type}}                    - exception type
    {{descriptionPlaceholder}}  - _description_ placeholder
{{/exceptions}}

{{#yields}}                     - iterate over yields
    {{typePlaceholder}}         - _type_ placeholder
    {{descriptionPlaceholder}}  - _description_ placeholder
{{/yields}}

{{#returns}}                    - iterate over returns
    {{typePlaceholder}}         - _type_ placeholder
    {{descriptionPlaceholder}}  - _description_ placeholder
{{/returns}}
```

# Additional Sections

```mustache
{{#argsExist}}                  - display contents if args exist
{{/argsExist}}

{{#kwargsExist}}                - display contents if kwargs exist
{{/kwargsExist}}

{{#parametersExist}}            - display contents if args or kwargs exist
{{/parametersExist}}

{{#exceptionsExist}}            - display contents if exceptions exist
{{/exceptionsExist}}

{{#yieldsExist}}                - display contents if returns exist
{{/yieldsExist}}

{{#returnsExist}}               - display contents if returns exist
{{/returnsExist}}

{{#placeholder}}                - makes contents a placeholder
{{/placeholder}}
```
#
