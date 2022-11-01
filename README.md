# QuPath tiling python

# Installation
`mothi` can be installed via `pip`:
```bash
 pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mothi
```

# Install via Docker
To get a ready to use (python, mothi and QuPath installed) Docker container,
use the [Dockerfile from github](https://github.com/thkauer/GBM_QuPath_tiles/blob/master/Dockerfile).  
Copy the Dockerfile to your local system and follow the below steps for creating an image:

```bash
  user@computer:~$ docker build [-t tagname] /path/to/Dockerfile
```

now run the container using:
```bash
  user@computer:~$ docker run -it tagname /bin/bash
```

To mount files or data, explore the [docker run documentation](https://docs.docker.com/engine/reference/commandline/run/) 


## Visual Studio Code Dev Container

If you want to use a `VSCode Dev Container`, copy the `Dockerfile` to the directory you want to work with.
Open the folder in VSCode and follow the VSCode Dev Container
[Guide](https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container) 


# For development
after cloning the repository follow the steps below to generate the documentation as html files.
```bash
cd repo/docs
make html
``` 
after building the documentation files, they can be found under `repo/docs/build/html`