============
Installation
============

Install mothi
-------------

| To install `mothi` via `pypi`, run (only `test-pypi` yet):

.. code-block:: console

    user@computer: ~$ pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mothi


Install QuPath
--------------

| To interact with `QuPath`, `paquo (package that mothi extends)` requires a working `QuPath` installation.
  To install `QuPath` follow the `QuPath` installation guide:
  `Install QuPath <https://qupath.readthedocs.io/en/stable/docs/intro/installation.html>`_.
| If `QuPath` is not installed in the default directory, you need to configure `QuPath` for `paquo` via:

.. code-block:: console

  use via enviroment variable
  user@computer: ~$ export PAQUO_QUPATH_DIR=/path/to/QuPath

| or via the `configuration <https://paquo.readthedocs.io/en/latest/configuration.html#configuration>`_
  of `paquo`