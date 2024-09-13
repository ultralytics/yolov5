#!C:\Users\Anakin\Desktop\school stuff\MDP\yolov5\yolo-env\Scripts\python.exe

# $Id: rst2s5.py 4564 2006-05-21 20:44:42Z wiemann $
# Author: Chris Liechti <cliechti@gmx.net>
# Copyright: This module has been placed in the public domain.
"""A minimal front end to the Docutils Publisher, producing HTML slides using the S5 template system."""

try:
    import locale

    locale.setlocale(locale.LC_ALL, "")
except:
    pass

from docutils.core import default_description, publish_cmdline

description = (
    "Generates S5 (X)HTML slideshow documents from standalone " "reStructuredText sources.  " + default_description
)

publish_cmdline(writer_name="s5", description=description)
