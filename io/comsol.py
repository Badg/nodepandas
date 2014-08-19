import pandas as pd
import numpy as np

# Grab the io dependencies
from . import io
from .io import *

from .. import core
from ..core import *

# Should I add a bit of logging or summat? I had it in dummy3.py

def load_mphtxt(mphfile, infer_lines=50):
    """Loads a comsol text results file. Attempts to auto-detect fields, and
    upon failure, assigns arbitrary field names."""

    previewlines = []
    inrow = 'dummy'
    previewdata = []

    with open(mphfile, 'r', newline='') as txtFile:
        # Prep the file for pandas conversion
        header_line_nums, headers, fields, units = \
            mphtxt_prep_headers(txtFile)

        ii = 0
        while inrow and len(previewlines) < infer_lines:
            inrow = txtFile.readline()

            # If the current line is not a header, append it
            if ii not in header_line_nums:
                previewlines.append(inrow)

            # Increment the line number
            ii += 1

    colspecs = mphtxt_detect_colspec(previewlines)

    # Convert the preview data into a 2d list of strings
    for line in previewlines:
        dothesplit = []
        for colspec in colspecs:
            dothesplit.append(line[colspec[0]:colspec[1]].strip())
        previewdata.append(dothesplit)

    # if fields:
    #     # If we've reached this point, we've detected fields.
    #     print( "\n####### Autodetected the following fields: ")
    #     print( "============================================\n")
    #     print("Note that these fields are NOT case sensitive. Duplicates will"
    #         "\nbe eliminated, possibly resulting in the wrong label being "
    #         "\napplied to data.  Please rename any non-unique fields "
    #         "\nexplicitly.\n")
    #     for field in fields:
    #         print("    + " + field)

    # enterFields = slui.query_confirm("Would you like to enter or correct "
    #     "fields?  Missing fields \nwill be padded with generic"
    #     "(field1, field2, field3...) \n labels.", default='no')
    # if enterFields:
    #     fields = None
    #     while not fields:
    #         fields = slui.query_string("Go for it!", True)

    # Pad remaining fields
    io.pad_fields(fields, colspecs)

    return header_line_nums, headers, colspecs, previewdata, fields, units

def import_mphtxt(mphfile, headerlines, colspecs, fields, units=None):
    # Perform the conversion
    rawdf = spacetxt_to_panda(mphfile, headerlines, colspecs, fields)
    # Round to a bit-more-than-reasonable (given numerical accuracy) digit
    rawdf=np.round(rawdf, 8)
    # Remove duplicate rows, since comsol is a dick
    rawdf.drop_duplicates(inplace=True)

    return rawdf

def mphtxt_prep_headers(txtFile):
    """ Extracts the header from a mphtxt file, generates column names, etc.

    Does some reasonably intelligent pre-processing of the txt file and then 
    generates an output tuple including the headers (for logging), the numbers 
    of header lines (to skip during pandas conversion), the column specs (for 
    pandas import), the column names, etc.  See below.
    
    Inputs:
    -------
    + <txtFilename> is the path to the COMSOL text file.  If left blank, it
      will query the user.
    + <infer_lines> is the number of lines to use to infer the column widths.
    + <basename> is the "root" name to apply to all of the set.  If left 
      blank, it will be inferred from <txtFilename>.  As an example of this 
      inference, "foo.txt" would become a <basename> of "foo", which would 
      then turn into files like foo_log.txt or foo_raw.h5.  Will append 
      generic fieldnames to <fields> if the number of <fields> is less than
      the number of detected columns.
    
    Outputs (as tuple, in this order):
    + <basename>, a str of the (inferred or otherwise) base name
    + <headerlines>, a list of the int line numbers of headers to skip parsing
    + <headers>, a list (the elements are str and include \n) of strs
    + <fields>, a list of strs for column labels
    + <colspecs>, a list of tuples of column start -> ends in panda form (ex:
      [(0,3),(4,7)], etc)    
    """    
    # Localglobal declarations
    import re
    
    fields = []
    units = []
    rawfields = []
    notDoGroup = None
    group = None
    groupname = None
    inrow = 'dummy'
    txtheader = ""
    headerlines = []
    headers = []
    colspec = []
    ii = -1
    # leakCount = 0

    header = True
    # Parse each line until readline returns '', signalling end of file
    while inrow: 
        # Always, always increment the row count, regardless of behavior
        ii += 1
        
        inrow = txtFile.readline()
        # Skip blank rows
        if not inrow or not inrow[0]: continue 
        
        # Look to see if it's a header
        header, payload = detect_columns(inrow)
        
        if header:
            # It's a header row
            headerlines.append(ii)
            if isinstance(payload, str):
                # Immediately pass the row to the header variable
                headers.append(payload)
                # We're done here.
                continue

            # Now we definitely have fields.  Make them lower than an
            # alcoholic in a clown suit pissing off the side of the
            # golden gate bridge.  And uniquer, too.
            if isinstance(payload, list):
                #rawfields = remove_duplicates([fld.lower() for fld in payload])
                rawfields = remove_duplicates(payload)
        else:
            break
    #        leakCount += 1

    # Manipulate fields and generate units
    # Generate a regex that will group the field and units from comsol's mess
    p = re.compile(r'\((?P<field>.*)\)\[1/\((?P<units>.*)\)\]')
    for rawfield in rawfields:
        # Look for matching strings
        m = p.search(rawfield)
        # If this produced a match, work with it:
        if m:
            fields.append(m.group('field'))
            units.append(m.group('units'))
        # If no match, append the raw string as a label and empty as units
        else:
            fields.append(rawfield)
            units.append('')

    # Return the file position to zero it was:
    txtFile.seek(0)

    # Now we should know how many rows of headers we have and what the 
    # fields are called.
    return headerlines, headers, fields, units

def mphtxt_detect_colspec(lines):
    # Since it appears that pandas doesn't do a good job of 
    # automagic column detection, let's go ahead and do that here.
    # All headers have been parsed.  This should be data.
    # Note that the file pointer must already be advanced to the start of the
    # data for this to work.
    from statistics import mean
    widths = []

    for line in lines:
        # Get the raw column widths
        thiswd = detect_columns(line, returnwidths=True)
        # Replace the irregular short width of the last column with 
        # the max colwidth seen in the row (including the last column)
        thiswd[len(thiswd)-1] = max(thiswd)
        widths.append(thiswd)
        continue

    # Let's examine each set of widths by transposing them first:
    widths = zip(*widths)
    cols = []
    # Now iterate on each field column
    for field in widths:
        # Append the rounded average length to the cols list
        colavg = mean(field)
        cols.append(int(round(colavg)))
        
    # Cols now contains the calculated average column widths. The ends will be
    # width - 1... fuckit, anyway point is now we're going to generate the
    # colspec for pandas read_fwf
    colspecs = []
    offset = -1
    for it in cols:
        colstarts = offset + 1
        offset += it
        colends = offset
        colspecs.append([colstarts, colends])

    return colspecs