import sys, copy, numpy as np
sys.path.append('../../')

import csv
from py_utils.utils import checkEqual, is_num

def csv_to_row_dicts(filename, keys=[], display=False, row_limit=0):
    data = {};  record_idx = 0
    with open(filename, 'r') as infile:
        csvreader = csv.reader(infile)
        headers = map(lambda x: x.lower(), csvreader.next())
        if display:
            print "Headers: %s" % sorted(headers)

        for idx, row in enumerate(csvreader):
            keyval = '' if keys else "%s" % record_idx
            record_idx += 1
            rowdict = {}
            for h, v in zip(headers, row):
                rowdict[h] = v
                if keys and h in keys:
                    keyval += v
            data[keyval] = rowdict

            if row_limit and idx > row_limit:
                break

    if display:
        print "%s records in %s" % (len(data), filename)

    return data, headers


def remove_ignored(data, ignored=[], target_field=''):
    for entry in data:
        for field in ignored:
            if field in entry:
                del entry[field]


def get_targets(data, target_field, balance=False, split_multiclass=False):
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
    targets = []; target_dict = {'Isurehopethiswillneverbearealclass': -1.0}
    for entry in data:
        if target_field and target_field in entry:
            value = entry[target_field]
            # 2-class numeric targets
            if is_num(value):
                targets.append(float(value))
            # Multi-class and text-field targets
            else:
                if value not in target_dict:
                    target_dict[value] = max(target_dict.values()) + 1.0

                targets.append(target_dict[value])

            del entry[target_field]

    return data, np.array(targets)

def split_multiclass(targets):
    target_dict = {'Isurehopethiswillneverbearealclass': -1.0}
    for value in targets:
        if value not in target_dict:
            target_dict[value] = max(target_dict.values()) + 1.0

    target_matrix = []
    L = int(max(target_dict.values()) + 1)
    for value in targets:
        row = [0.0] * L
        row[int(value)] = 1.0
        target_matrix.append(row)

    return np.array(target_matrix)