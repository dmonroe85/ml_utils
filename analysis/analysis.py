import re, csv

def compare_data(dict1, dict2, numeric_precision=2, display=False):
    matched_data = {}
    matches = 0; misses = 0
    missing_fields = []
    for key in dict1:
        if key in dict2:
            miss = False
            pdata1 = dict1[key]
            pdata2 = dict2[key]
            for field in pdata1:
                if field in pdata2:
                    match = False
                    # Limit the precision; checks will fail because of precision differences
                    #   TODO: Need a way to set the precision
                    try:
                        match = \
                            ("%.*f%%" % (numeric_precision, float(pdata1[field]))) == \
                            ("%.*f%%" % (numeric_precision, float(pdata2[field])))
                    except:
                        match = pdata1[field] == pdata2[field]

                    # Print the field that failed the check
                    if not match:
                        print field
                        miss = True

                elif field not in missing_fields:
                    missing_fields.append(field)
            if miss:
                misses += 1
                print pdata1
                print pdata2
            # Count and store the matched data into a new dictionary
            else:
                matches += 1
                matched_data[key] = pdata1

    if display:
        print "\nMatches: %s, Misses: %s" % (matches, misses)
        print "Length of set 1: %s" % len(dict1.keys())
        print "Length of set 2: %s" % len(dict2.keys())
        print "Missing Fields: %s" % missing_fields

    return matched_data

def val_frequency_hist(data):
    """ Value frequency histogram generator:

    This parses data produced with csv_to_row_dict (list of dictionaries).  It
    summarizes all of the contained fields and the frequency of each unique
    value within those fields.
    """
    histogram = {}
    for entry in data:
        for field in entry:
            if field not in histogram:
                histogram[field] = {entry[field]: 1.0}
            elif entry[field] not in histogram[field]:
                histogram[field][entry[field]] = 1.0
            else:
                histogram[field][entry[field]] += 1.0
    return histogram

def write_val_hist(histogram, print_fields=[], filename=''):
    """ Value frequency histogram writer:

    The default behavior prints out each field recorded in the histogram and the
    number of unique values listed therein.

        If field are specified in "print_fields", each value and their
            occurrences will also be listed.

        If filename is specified, this information will be written to CSV;
            otherwise, it will be printed to STDout.
    """
    if filename:
        outfile = open(filename, 'wb')
        csv_writer = csv.writer(outfile)

    for field in histogram:
        if filename:
            csv_writer.writerow([field, len(histogram[field])])
        else:
            print "%s, %s" % (field, len(histogram[field]))

        if field in print_fields:
            for v in histogram[field]:
                if filename:
                    csv_writer.writerow(['', v, histogram[field][v]])
                else:
                    print "  %s, %s" % (v, histogram[field][v])

    if filename:
        outfile.close()

def analyze_date_format(data, date_field):
    """ Date format analyzer:

    Finds the date field and replaces numeric and character values with
    arbitrary characters and counts their unique occurrences.
    """
    date_formats = {}
    for entry in data:
        for field in entry:
            if field == date_field:
                number_format = re.sub(r'\d', '#', entry[field])
                alpha_format  = re.sub(r'[A-Za-z]', 'A', number_format)
                if alpha_format not in date_formats:
                    date_formats[alpha_format] = 1
                else:
                    date_formats[alpha_format] += 1
    return date_formats