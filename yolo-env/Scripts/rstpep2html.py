#!C:\Users\Anakin\Desktop\school stuff\MDP\yolov5\yolo-env\Scripts\python.exe

# $Id: rstpep2html.py 4564 2006-05-21 20:44:42Z wiemann $
# Author: David Goodger <goodger@python.org>
# Copyright: This module has been placed in the public domain.
"""A minimal front end to the Docutils Publisher, producing HTML from PEP (Python Enhancement Proposal) documents."""

try:
    import locale

    locale.setlocale(locale.LC_ALL, "")
except:
    pass

from docutils.core import default_description, publish_cmdline

description = "Generates (X)HTML from reStructuredText-format PEP files.  " + default_description

publish_cmdline(reader_name="pep", writer_name="pep_html", description=description)
