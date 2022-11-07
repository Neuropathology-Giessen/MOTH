==========
Quickstart
==========

| All non tile specifig QuPath things are implemented and documented by `paquo`.
  Please refer to paquo's `documentation <https://paquo.readthedocs.io/en/latest/index.html>`_
  for these functionalities.
| The focus of the package is on the use of tiles in QuPath, for example to enable a pytorch workflow.
  To get started with QuPath tiling in `python`, here are a few examples of how to use `mothi`:

-------------------------------
Get tiles and their annotations
-------------------------------

| The first use case of mothi is to query specific tiles and the associated annotations.
| Below is small example of how to use `mothi` to get theese tiles and their
  associated annotations.

Open a project to work on it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| The functions of `mothi` become usable via the :class:`mothi.tiling_projects.QuPathTilingProject`
  class. 

.. code-block:: python3

    >>> from mothi.tiling_projects import QuPathTilingProject
    >>> qp_project = QuPathTilingProject('/path/to/project')

| If a valid path was specified, the project is now opened in read only mode.

Get tile and its annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| To retrieve tiles and their annotations just call
  :meth:`mothi.tiling_projects.QuPathTilingProject.get_tile` 
  and :meth:`mothi.tiling_projects.QuPathTilingProject.get_tile_annot_mask`
  methods with the desired parameters and the tile and its annotations will be returned

.. code-block:: python3

    >>> tile = qp_project.get_tile(img_id=0, location=(50,50), size=(256,256))
    >>> tilemask = qp_project.get_tile_annot_mask(img_id=0, location=(50,50), size=(256,256))

| The example shown above returns tiles and annotations for the first image at position
  (50|50) in size 256 x 256 pixels.
| Learn more about the parameters of the functions by taking a look at the :ref:`mothi api`.

---------------------------
Save a tilemask on an image
---------------------------
| The second use case of mothi is storing generated annotations (tilemasks) on images.

Open a project to work on it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| To save annotations on images an existing project must be opened in non-read-only mode
  or a new one must be created

.. code-block:: python3

    >>> # example: open in read/write mode
    >>> from mothi.tiling_projects import QuPathTilingProject
    >>> qp_project = QuPathTilingProject('/path/to/project', mode='r+')

.. code-block:: python3

    >>> # create new project
    >>> from mothi.tiling_projects import QuPathTilingProject
    >>> qp_project = QuPathTilingProject('/path/to/project', mode='x')

Save tilemask
~~~~~~~~~~~~~
| The `tilemask` you want to save can now be saved by calling the method
  :meth:`mothi.tiling_projects.QuPathTilingProject.save_mask_annotations`

.. code-block:: python3

    >>> qp_project.save_mask_annotations(img_id=0, annot_mask=tilemask, location=(50,50))

| The example will save the generated `tilemask` in the first image 
  starting at (50|50).
| Learn more about the parameters of the function by taking a look at the :ref:`mothi api`.

Merge annotation
~~~~~~~~~~~~~~~~
| After importing multiple tile annotations, you can merge nearby annotations of the same classes.
  This can be done with the help of the method
  :meth:`mothi.tiling_projects.QuPathTilingProject.merge_near_annotations`.

.. code-block:: python3

    >>> qp_project.merge_near_annotations(img_id=0, max_dist=0)

| This will merge all neighboring annotations that have the same class and no spacing
  in the first image.