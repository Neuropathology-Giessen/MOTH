# QuPath tiling python

# Installation
`mothi` can be installed via `pip`:
```bash
 user@computer:~$ pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mothi
```

## Install QuPath
To interact with QuPath, paquo (package that mothi extends) requires a working QuPath installation.  
To install QuPath follow the QuPath installation guide:
  [Install QuPath](https://qupath.readthedocs.io/en/stable/docs/intro/installation.html).  

If `QuPath` is not installed in the default directory, you need to configure QuPath for paquo via:

```bash
  user@computer:~$ export PAQUO_QUPATH_DIR=/path/to/QuPath
```

or via the [configuration](https://paquo.readthedocs.io/en/latest/configuration.html#configuration) of paquo.

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
after cloning the repository create a `VSCode Dev Container` with the DevDockerfile.
In the Container, open the repository and follow the steps below:  

```bash
  user@computer:~/path/to/repo$ cd ./docs
  user@computer:~/path/to/repo/docs$ make html
``` 

after building the documentation files, they can be found under `repo/docs/build/html`