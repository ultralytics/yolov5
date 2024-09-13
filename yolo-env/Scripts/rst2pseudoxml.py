#!C:\Users\Anakin\Desktop\school stuff\MDP\yolov5\yolo-env\Scripts\python.exe

# $Id: rst2pseudoxml.py 4564 2006-05-21 20:44:42Z wiemann $
# Author: David Goodger <goodger@python.org>
# Copyright: This module has been placed in the public domain.
"""A minimal front end to the Docutils Publisher, producing pseudo-XML."""

try:
    import locale

    locale.setlocale(locale.LC_ALL, "")
except:
    pass

from docutils.core import default_description, publish_cmdline

description = (
    "Generates pseudo-XML from standalone reStructuredText " "sources (for testing purposes).  " + default_description
)

publish_cmdline(description=description)
