============
Installation
============

Install MOTH
-------------

| To install `moth` via `pypi`, run (only `test-pypi` yet):

.. code-block:: console

    user@computer:~$ pip install gin-moth


Install QuPath
--------------

| To interact with `QuPath`, `paquo (a package that moth extends)` requires a working `QuPath` installation.
  To install `QuPath` follow the `QuPath` installation guide:
  `Install QuPath <https://qupath.readthedocs.io/en/stable/docs/intro/installation.html>`_.
| If `QuPath` is not installed in the default directory, you need to configure `QuPath` for `paquo` via:

.. code-block:: console

  use via enviroment variable
  user@computer:~$ export PAQUO_QUPATH_DIR=/path/to/QuPath

| or via the `configuration <https://paquo.readthedocs.io/en/latest/configuration.html#configuration>`_
  of `paquo`


Install via Docker
------------------

| To get a ready to use (python, moth, and QuPath installed) Docker container,
  clone the repository and
  use the `Dockerfile from github <https://github.com/thkauer/GBM_QuPath_tiles/blob/master/Dockerfile>`_ 
  to create a new Docker image.
| To use the Dockerfile follow the below steps for creating an image:

.. code-block:: console

  user@computer:~/path/to/the/repository$ docker build [-t tagname] .


| now you can run the container using:

.. code-block:: console

  user@computer:~/path/to/the/repository$ docker run -it tagname bash

| To mount files or data, explore the 
  `docker run documentation <https://docs.docker.com/engine/reference/commandline/run/>`_ 


Visual Studio Code Dev Container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

| If you want to use a VSCode `Dev Container`, copy the `Dockerfile` to the directory you want to work with.
  Open the folder in VSCode and follow the
  `VSCode Dev Container Guide <https://code.visualstudio.com/docs/devcontainers/containers#_quick-start-open-an-existing-folder-in-a-container>`_ 