import sys, copy, numpy as np
sys.path.append('../../')

import csv
from py_utils.utils import checkEqual

def csv_to_row_dicts(filename, keys, display=False):
    data = {}
    with open(filename, 'r') as infile:
        csvreader = csv.reader(infile)
        headers = map(lambda x: x.lower(), csvreader.next())
        if display:
            print "Headers: %s" % sorted(headers)

        for row in csvreader:
            keyval = ''; rowdict = {}
            for h, v in zip(headers, row):
                rowdict[h] = v
                if h in keys:
                    keyval += v
            data[keyval] = rowdict
    return data, headers

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

def remove_ignored(data, ignored=[], target_field=''):
    for entry in data:
        for field in ignored:
            if field in entry:
                del entry[field]

def get_targets(data, target_field, balance=False):
    if balance:
        # Second Pass, get target statistics
        target_groups = {}
        for entry in data:
            if target_field and target_field in entry:            
                # Balancing Classes
                target_val = entry[target_field]
                if target_val not in target_groups:
                    target_groups[target_val] = []

                target_groups[target_val].append(entry)

        # Resampling to balance out the classes
        while not checkEqual(map(lambda x: len(target_groups[x]), target_groups)):
            class_summary = map(lambda x: [x, len(target_groups[x])], target_groups)
            min_target = sorted(class_summary, key=lambda x: x[1])[0][0]

            target_groups[min_target].append(
                copy.deepcopy(
                    np.random.choice(
                        target_groups[min_target]
                    )
                )
            )

        # Add the groups back to the original data
        new_data = []
        for t in target_groups:
            new_data += copy.deepcopy(target_groups[t])

        data = new_data

    # generate the array of targets
    targets = []
    for entry in data:
        if target_field and target_field in entry:
            targets.append(int(entry[target_field]))
            del entry[target_field]

    return data, np.array(targets)