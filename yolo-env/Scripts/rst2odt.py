#!C:\Users\Anakin\Desktop\school stuff\MDP\yolov5\yolo-env\Scripts\python.exe

# $Id: rst2odt.py 5839 2009-01-07 19:09:28Z dkuhlman $
# Author: Dave Kuhlman <dkuhlman@rexx.com>
# Copyright: This module has been placed in the public domain.
"""A front end to the Docutils Publisher, producing OpenOffice documents."""

try:
    import locale

    locale.setlocale(locale.LC_ALL, "")
except:
    pass

from docutils.core import default_description, publish_cmdline_to_binary
from docutils.writers.odf_odt import Reader, Writer

description = (
    "Generates OpenDocument/OpenOffice/ODF documents from "
    "standalone reStructuredText sources.  " + default_description
)


writer = Writer()
reader = Reader()
output = publish_cmdline_to_binary(reader=reader, writer=writer, description=description)
