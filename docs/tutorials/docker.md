# Docker

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
