import sys
sys.path.append('../../')

from py_utils.utils import is_num
from ml_utils.learners.collab_filter import cofi_ls

from sklearn import feature_extraction
import numpy as np
import copy

class PercentileCategorizer(object):
    def __init__(self, features={}):
        self.percentiles = {}
        self.set_params(features)

    def get_params(self, deep):
        """ Needed for the sklearn pipeline.
        """
        return {'features': self.features}

    def set_params(self, features={}):
        """ Needed for the sklearn pipeline.
        """
        self.features = features

    def fit(self, data, targets=[]):
        numeric_fields = {}
        for row in data:
            for field in self.features:
                if field in row and is_num(row[field]):
                    if field not in numeric_fields:
                        numeric_fields[field] = []

                    numeric_fields[field].append(float(row[field]))
        
        for field in self.features:
            if field in self.percentiles:
                self.percentiles[field] = stats.mstats.mquantiles(
                    numeric_fields[field],
                    prob=[float(x)/float(self.features[field]) for x in range(1, self.features[field] + 1)]
                )

    def transform(self, data, targets=[]):
        new_data = []
        # Iterate through rows...
        for row in data:
            new_row = copy.deepcopy(row)
            # Stated Numeric Features...
            for field in self.percentiles:
                # If the current field is in the row and it is non empty...
                if field in new_row and new_row[field] and is_num(new_row[field]):
                    # Locate the correct percentile.
                    current_val = float(new_row[field])
                    new_row[field] = current_val
                    for p in self.percentiles[field]:
                        if current_val <= p:
                            new_row[field + '_percentile'] = "%s" % p
                            break
                    else:
                        new_row[field + '_percentile'] = "%s" % p
                else:
                    new_row[field + '_percentile'] = ""

            new_data.append(new_row)

        return new_data

    def fit_transform(self, data, targets=[]):
        self.fit(data)
        return self.transform(data)

class LowCountTrimmer(object):
    def __init__(self, threshold=1, criteria='field'):
        self.set_params(threshold, criteria)
        self.remove_fields = []

    def get_params(self, deep):
        """ Needed for the sklearn pipeline.
        """
        return {'threshold': self.threshold, 'criteria': self.criteria}

    def set_params(self, threshold=1, criteria='field'):
        """ Needed for the sklearn pipeline.
        """
        self.threshold = threshold
        self.criteria = criteria

    def fit(self, data, targets=[]):
        counts = {}; N = 0
        for row in data:
            N += 1
            for field in row:
                if self.criteria == 'field':
                    key = field
                elif self.criteria == 'value':
                    key = field + '_%s' % row[field]

                if key not in counts or not counts[key]:
                    counts[key] = 0

                counts[key] += 1

        if self.threshold >= N:
            self.threshold = N-1

        count_keys = set(map(lambda x: x.split('_')[0], counts))
        for key in counts:
            if self.criteria == 'field':
                if counts[key] <= self.threshold:
                    self.remove_fields.append(key)

            elif self.criteria == 'value':
                if counts[key] > self.threshold:
                    field_name = key.split('_')[0]
                    if field_name in count_keys:
                        count_keys.remove(field_name)

        if self.criteria == 'value':
            self.remove_fields = list(count_keys)


    def transform(self, data, targets=[]):
        new_data = []
        for row in data:
            new_row = copy.deepcopy(row)
            for field in self.remove_fields:
                if field in new_row:
                    del new_row[field]

            new_data.append(new_row)

        return new_data

    def fit_transform(self, data, targets=[]):
        self.fit(data)
        return self.transform(data)

class CollaborativeFilter(object):
    def __init__(self, L=0, F=10, iterations=100, verbose=0):
        self.set_params(L, F, iterations, verbose)

    def set_params(self, L=0, F=10, iterations=100, verbose=0):
        # Regularization Parameter
        self.L = L
        # Number of Feature Vectors
        self.F = F
        # Number of Iterations To Run
        self.iterations = iterations
        # Output verbosity
        self.verbose = verbose

    def get_params(self, deep=False):
        return {
            "L": self.L,
            "F": self.F,
            "iterations": self.iterations,
            "verbose": self.verbose,
        }

    def fit(self, data, targets=[]):
        pass

    def transform(self, data, targets=[]):
        DV = feature_extraction.DictVectorizer(sparse=False)
        Y = DV.fit_transform(data)

        # Get the vocabulary from the transformation
        sorted_vocab = sorted(DV.vocabulary_.items(), key=lambda x: x[1])
        field_names = map(lambda x: x[0], sorted_vocab)
        N_fields = len(field_names)

        # Empty Indicators
        empty_indicators = filter(
            lambda x: '=' in x and x.split('=')[1] == '',
            field_names
        )

        # Create an array mask indicating the locations of the "empty"
        # indicators in an input row.
        detector = np.array(map(
            lambda x: (1.0 if '=' in x and x.split('=')[1] == '' else 0.0),
            field_names
        ))

        mask_rows = []
        for field1 in field_names:
            if field1.endswith('='):
                mask_row = []
                for field2 in field_names:
                    if field2.split('=')[0] + '=' == field1 and field2 != field1:
                        mask_row.append(1.0)
                    else:
                        mask_row.append(0.0)
            else:
                mask_row = [0.0] * N_fields

            mask_rows.append(mask_row)
        mask_matrix = np.array(mask_rows).T

        # Find the values of R by locating entries where the empty indicators
        # are set to 1.
        if self.verbose >= 1:
            print "Finding R"
        R = []
        for idx, y in enumerate(Y):
            if not idx % 100:
                if self.verbose >= 2:
                    print "Evaluated %s rows" % idx

            # Find if any misses have been indicated in this row
            miss_flags = np.logical_and(y, detector)
            R.append(mask_matrix.dot(miss_flags))

        R = np.logical_not(np.array(R)).astype(float)

        # Now we are ready to estimate using the collaborative filtering
        # algorithm.
        if self.verbose >= 1:
            print "Filtering Collaboratively"
        X, T, P = cofi_ls(Y, R,
            L=self.L, F=self.F, iterations=self.iterations)

        Thresholded = []
        for field, column in zip(field_names, P.T):
            if '=' in field:
                Thresholded.append((column >= 0.5).astype(float))
            else:
                Thresholded.append(column)

        Filled_in = R*Y + np.logical_not(R)*np.array(Thresholded).T

        # Debugging Purposes...
        # import csv
        # with open('test_file.csv', 'wb') as outfile:
        #     csvout = csv.writer(outfile)
        #     csvout.writerow(field_names)
        #     csvout.writerows(Y)
        #     csvout.writerow([])
        #     csvout.writerows(R)
        #     csvout.writerow([])
        #     csvout.writerows(Filled_in)
        #     csvout.writerow([])
        #     csvout.writerows(P)
        #     csvout.writerow([])
        #     csvout.writerows(X.dot(T.T))

        return DV.inverse_transform(Filled_in)

    def fit_transform(self, data, targets=[]):
        self.fit(data, targets)
        return self.transform(data, targets)
