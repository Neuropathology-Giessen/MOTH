.. _quickstart:

==========
Quickstart
==========

| All non tile specific QuPath things are implemented and documented by `paquo`.
  Please refer to paquo's `documentation <https://paquo.readthedocs.io/en/latest/index.html>`_
  for these functionalities.
| The focus of the package is on the use of tiles in QuPath, for example to enable a pytorch workflow.
  To get started with QuPath tiling in `python`, here are a few examples of how to use `moth`:

-------------------------------
Get tiles and their annotations
-------------------------------

| The first use case of moth is to query specific tiles and the associated annotations.
| Below is a small example of using `moth` to get the tiles and their annotations.

Open a project to work on it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| The functions of `moth` become usable via the :class:`moth.projects.QuPathTilingProject`
  class. 

.. code-block:: python3

    >>> from moth import QuPathTilingProject, MaskParameter
    >>> qp_project = QuPathTilingProject('/path/to/project')

| If a valid path was specified, the project is now opened in read only mode.

Get tile and its annotations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| To retrieve tiles and their annotations just call
  :meth:`moth.projects.QuPathTilingProject.get_tile` 
  and :meth:`moth.projects.QuPathTilingProject.get_tile_annotation_mask`
  methods with the desired parameters and the tile and its annotations will be returned

.. code-block:: python3

    >>> tile = qp_project.get_tile(img_id=0, location=(50,50), size=(256,256))
    >>> tilemask = qp_project.get_tile_annotation_mask(MaskParameter(img_id=0, location=(50,50)), size=(256,256))

| The example shown above returns tiles and annotations for the first image at position
  (50|50) in size 256 x 256 pixels.
| Learn more about the parameters of the functions by taking a look at the :ref:`api`.

---------------------------
Save a tilemask on an image
---------------------------
| The second use case of moth is storing generated annotations (tilemask) on images.

Open a project to work on it
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
| To save annotations on images an existing project must be opened in non-read-only mode
  or a new one must be created

.. code-block:: python3

    >>> # example: open in read/write mode
    >>> from moth import QuPathTilingProject
    >>> qp_project = QuPathTilingProject('/path/to/project', mode='r+')

.. code-block:: python3

    >>> # create new project
    >>> from moth import QuPathTilingProject
    >>> qp_project = QuPathTilingProject('/path/to/project', mode='x')

Save tilemask
~~~~~~~~~~~~~
| The `tilemask` you want to save can now be saved by calling the method
  :meth:`moth.projects.QuPathTilingProject.save_mask_annotations`

.. code-block:: python3

    >>> qp_project.save_mask_annotations(annotation_mask=tilemask, MaskParameter(img_id=0, location=(50,50)))

| The example will save the generated `tilemask` in the first image 
  starting at (50|50).
| Learn more about the parameters of the function by taking a look at the :ref:`api`.

Merge annotation
~~~~~~~~~~~~~~~~
| After importing multiple tile annotations, you can merge nearby annotations of the same classes.
  This can be done with the help of the method
  :meth:`moth.projects.QuPathTilingProject.merge_near_annotations`.

.. code-block:: python3

    >>> qp_project.merge_near_annotations(img_id=0, max_dist=0)

| This will merge all neighboring annotations that have the same class and no spacing
  in the first image.