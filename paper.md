---
title: 'working on title'
tags:
  - Python
  - whole slide images
  - neuropathology
  - u-net
  - artificial intelligence
authors:
  - name: Thomas Kauer
    affiliation: 1
  - name: Jannik Sehring
    affiliation: 1
  - name: Till Acker
    affiliation: 1
  - name: Daniel Amsel
    corresponding: true
    affiliation: 1
    orcid: 0000-0002-0512-9802
affiliations:
 - name: Institute of Neuropathology, Justus Liebig University Giessen, Arndtstr. 16 D-35392 Giessen, Germany
   index: 1
date: 29 September 2022
bibliography: paper.bib



# Summary
Through the digitization of patient tissues new approaches in histological research are possible. And with the help of tools like QuPath, the potential of digital pathology can be used by using QuPath to start new approaches with Deep Learning. Segmentation tasks can be trained to decrease the time needed to work on one tissue and create new data for a wider analysis on tissues. To use this potential MOTH helps researchers and developers to integrate QuPath in the Environment of Python.

# Statement of need
The use of digital pathology has the potential to remarkably boost and enhance pathological workflows. With the digitization of patient tissue samples in the form of digital whole slide images (WSIs), new opportunities in the field of research, diagnostic and teaching appeared. (1)
WSIs are high resolution tissue images with a multi gigabyte size. This size makes it difficult to use WSIs directly for Deep Learning. A solution for this problem is patch extraction/ Tiling. Patch Extraction enables the training of deep learning networks on single regions of the whole Image, mostly with sizes between 32 x 32 pixels up to 10,000 x 10,000 pixel. (1, more)
For the interaction with such big WSIs the Java tool QuPath was developed (3). Beside the functionality to view and annotate WSIs, you can perform analysis with QuPath. With the Groovy Scripting editor integrated in QuPath also custom Scripts, for example running patch extraction, can be performed.
MOTH brings the patch extraction / tiling for data annotated in QuPath to python. Deep learning models can now be trained with simple python method calls. No Groovy scripting needed to generate patches with annotation masks. Further advantages are the opportunity to interact with tiles on the fly, integrating annotation masks in QuPath projects and merging nearby patch annotations of the same class together.
With those functions a whole python deep learning segmentation workflow can be established and the results can be visualized and discussed supported by QuPath.



# Citations


# Figures



# Acknowledgements
The authors thank the German Federal Ministry of Education and Research for funding of MIRACUM (BMBF FKZ 01ZZ1801) as well as the junior research group AI-RON (BMBF FKZ 01ZZ2017).

# References
