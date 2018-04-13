#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Module containing all criteria available for tests."""

import abc
import collections
import itertools
import math

import numpy as np


#: Minimum gain allowed for Local Search methods to continue searching.
EPSILON = 0.000001

#: Maximum rank allowed for sigma_j matrices in Conditional Inferente Tree framework
BIG_CONTINGENCY_TABLE_THRESHOLD = 200

#: Contains the information about a given split. When empty, defaults to
#: `(None, [], float('-inf'))`.
Split = collections.namedtuple('Split',
                               ['attrib_index',
                                'splits_values',
                                'criterion_value'])
Split.__new__.__defaults__ = (None, [], float('-inf'))


class Criterion(object):
    """Abstract base class for every criterion.
    """
    __metaclass__ = abc.ABCMeta

    name = ''

    @classmethod
    @abc.abstractmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the criterion.
        """
        # returns (separation_attrib_index, splits_values, criterion_value)
        pass



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       PC-ext                                              ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class PCExt(Criterion):
    name = 'PC-ext'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the PC-ext criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                contingency_table = tree_node.contingency_tables[attrib_index].contingency_table
                values_num_samples = tree_node.contingency_tables[
                    attrib_index].values_num_samples
                (new_contingency_table,
                 new_num_samples_per_value,
                 new_index_to_old) = cls._group_values(contingency_table, values_num_samples)
                principal_component = cls._get_principal_component(
                    len(tree_node.valid_samples_indices),
                    new_contingency_table,
                    new_num_samples_per_value)
                inner_product_results = np.dot(principal_component, new_contingency_table.T)
                new_indices_order = inner_product_results.argsort()

                best_gini = float('+inf')
                best_left_values = set()
                best_right_values = set()
                left_values = set()
                right_values = set(new_indices_order)
                for metaindex, first_right in enumerate(new_indices_order):
                    curr_split_impurity = cls._calculate_split_gini_index(
                        new_contingency_table,
                        new_num_samples_per_value,
                        left_values,
                        right_values)
                    if curr_split_impurity < best_gini:
                        best_gini = curr_split_impurity
                        best_left_values = set(left_values)
                        best_right_values = set(right_values)
                    if left_values: # extended splits
                        last_left = new_indices_order[metaindex - 1]
                        left_values.remove(last_left)
                        right_values.add(last_left)
                        right_values.remove(first_right)
                        left_values.add(first_right)
                        curr_ext_split_impurity = cls._calculate_split_gini_index(
                            new_contingency_table,
                            new_num_samples_per_value,
                            left_values,
                            right_values)
                        if curr_ext_split_impurity < best_gini:
                            best_gini = curr_ext_split_impurity
                            best_left_values = set(left_values)
                            best_right_values = set(right_values)
                        right_values.remove(last_left)
                        left_values.add(last_left)
                        left_values.remove(first_right)
                        right_values.add(first_right)
                    right_values.remove(first_right)
                    left_values.add(first_right)
                (best_left_old_values,
                 best_right_old_values) = cls._change_split_to_use_old_values(best_left_values,
                                                                              best_right_values,
                                                                              new_index_to_old)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[best_left_old_values, best_right_old_values],
                          criterion_value=best_gini))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_gini,
                 last_left_value,
                 first_right_value) = cls._gini_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_gini))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _gini_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_gini = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                gini_value = cls._get_gini_value(class_num_left,
                                                 class_num_right,
                                                 num_left_samples,
                                                 num_right_samples)
                if gini_value < best_gini:
                    best_gini = gini_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_gini, best_last_left_value, best_first_right_value)

    @staticmethod
    def _get_num_samples_per_side(values_num_samples, left_values, right_values):
        """Returns two sets, each containing the values of a split side."""
        num_left_samples = sum(values_num_samples[value] for value in left_values)
        num_right_samples = sum(values_num_samples[value] for value in right_values)
        return  num_left_samples, num_right_samples

    @staticmethod
    def _get_num_samples_per_class_in_values(contingency_table, values):
        """Returns a list, i-th entry contains the number of samples of class i."""
        num_classes = contingency_table.shape[1]
        num_samples_per_class = [0] * num_classes
        for value in values:
            for class_index in range(num_classes):
                num_samples_per_class[class_index] += contingency_table[
                    value, class_index]
        return num_samples_per_class

    @classmethod
    def _calculate_split_gini_index(cls, contingency_table, values_num_samples, left_values,
                                    right_values):
        """Calculates the weighted Gini index of a split."""
        num_left_samples, num_right_samples = cls._get_num_samples_per_side(
            values_num_samples, left_values, right_values)
        num_samples_per_class_left = cls._get_num_samples_per_class_in_values(
            contingency_table, left_values)
        num_samples_per_class_right = cls._get_num_samples_per_class_in_values(
            contingency_table, right_values)
        return cls._get_gini_value(num_samples_per_class_left, num_samples_per_class_right,
                                   num_left_samples, num_right_samples)

    @classmethod
    def _get_gini_value(cls, num_samples_per_class_left, num_samples_per_class_right,
                        num_left_samples, num_right_samples):
        """Calculates the weighted Gini index of a split."""
        num_samples = num_left_samples + num_right_samples
        left_gini = cls._calculate_node_gini_index(num_left_samples, num_samples_per_class_left)
        right_gini = cls._calculate_node_gini_index(num_right_samples, num_samples_per_class_right)
        return ((num_left_samples / num_samples) * left_gini +
                (num_right_samples / num_samples) * right_gini)

    @staticmethod
    def _calculate_node_gini_index(num_split_samples, num_samples_per_class_in_split):
        """Calculates the Gini index of a node."""
        if not num_split_samples:
            return 1.0
        gini_index = 1.0
        for curr_class_num_samples in num_samples_per_class_in_split:
            gini_index -= (curr_class_num_samples / num_split_samples)**2
        return gini_index

    @classmethod
    def _group_values(cls, contingency_table, values_num_samples):
        """Groups values that have the same class probability vector. Remove empty values."""
        (interm_to_orig_value_int,
         interm_contingency_table,
         interm_values_num_samples) = cls._remove_empty_values(contingency_table,
                                                               values_num_samples)
        prob_matrix_transposed = np.divide(interm_contingency_table.T, interm_values_num_samples)
        prob_matrix = prob_matrix_transposed.T
        row_order = np.lexsort(prob_matrix_transposed[::-1])
        compared_index = row_order[0]
        new_index_to_old = [[interm_to_orig_value_int[compared_index]]]
        for interm_index in row_order[1:]:
            if np.allclose(prob_matrix[compared_index], prob_matrix[interm_index]):
                new_index_to_old[-1].append(interm_to_orig_value_int[interm_index])
            else:
                compared_index = interm_index
                new_index_to_old.append([interm_to_orig_value_int[compared_index]])
        new_num_values = len(new_index_to_old)
        num_classes = interm_contingency_table.shape[1]
        new_contingency_table = np.zeros((new_num_values, num_classes), dtype=int)
        new_num_samples_per_value = np.zeros((new_num_values), dtype=int)
        for new_index, old_indices in enumerate(new_index_to_old):
            new_contingency_table[new_index] = np.sum(contingency_table[old_indices, :], axis=0)
            new_num_samples_per_value[new_index] = np.sum(values_num_samples[old_indices])
        return new_contingency_table, new_num_samples_per_value, new_index_to_old

    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @staticmethod
    def _change_split_to_use_old_values(new_left, new_right, new_index_to_old):
        """Change split values to use indices of original contingency table."""
        left_old_values = set()
        for new_index in new_left:
            left_old_values |= set(new_index_to_old[new_index])
        right_old_values = set()
        for new_index in new_right:
            right_old_values |= set(new_index_to_old[new_index])
        return left_old_values, right_old_values

    @classmethod
    def _get_principal_component(cls, num_samples, contingency_table, values_num_samples):
        """Returns the principal component of the weighted covariance matrix."""
        num_samples_per_class = cls._get_num_samples_per_class(contingency_table)
        avg_prob_per_class = np.divide(num_samples_per_class, num_samples)
        prob_matrix = contingency_table / values_num_samples[:, None]
        diff_prob_matrix = (prob_matrix - avg_prob_per_class).T
        weight_diff_prob = diff_prob_matrix * values_num_samples[None, :]
        weighted_squared_diff_prob_matrix = np.dot(weight_diff_prob, diff_prob_matrix.T)
        weighted_covariance_matrix = (1/(num_samples - 1)) * weighted_squared_diff_prob_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance_matrix)
        index_largest_eigenvalue = np.argmax(np.square(eigenvalues))
        return eigenvectors[:, index_largest_eigenvalue]

    @staticmethod
    def _get_num_samples_per_class(contingency_table):
        """Returns a list, i-th entry contains the number of samples of class i."""
        num_values, num_classes = contingency_table.shape
        num_samples_per_class = [0] * num_classes
        for value in range(num_values):
            for class_index in range(num_classes):
                num_samples_per_class[class_index] += contingency_table[value, class_index]
        return num_samples_per_class



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                     HYPERCUBE COVER                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################


class HypercubeCover(Criterion):
    """Hypercube Cover criterion."""
    name = 'Hypercube Cover'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the Hypercube Cover
        criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                best_children_gini_gain = float('+inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                for (set_left_classes,
                     set_right_classes) in cls._generate_superclasses(
                         tree_node.class_index_num_samples):
                    (superclass_contingency_table,
                     superclass_index_num_samples) = cls._get_superclass_contingency_table(
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         set_left_classes,
                         set_right_classes)
                    (curr_gini_gain,
                     left_values,
                     right_values) = cls._two_class_trick(
                         tree_node.class_index_num_samples,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         superclass_contingency_table,
                         len(tree_node.valid_samples_indices))

                    if curr_gini_gain < best_children_gini_gain:
                        best_children_gini_gain = curr_gini_gain
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[best_left_values, best_right_values],
                          criterion_value=best_children_gini_gain))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_gini,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_gini))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @staticmethod
    def _generate_superclasses(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_superclass_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                          set_right_classes):
        superclass_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                superclass_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                superclass_contingency_table[value][1] += contingency_table[value][class_index]
        return superclass_contingency_table, superclass_index_num_samples

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_gini = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                gini_value = cls._calculate_children_gini_index(num_left_samples,
                                                                class_num_left,
                                                                num_right_samples,
                                                                class_num_right)
                if gini_value < best_gini:
                    best_gini = gini_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_gini, best_last_left_value, best_first_right_value)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side / side_num) ** 2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index

    @classmethod
    def _two_class_trick(cls, class_index_num_samples, superclass_index_num_samples, values_seen,
                         contingency_table, values_num_samples, superclass_contingency_table,
                         num_total_valid_samples):
        # TESTED!
        def _get_non_empty_superclass_indices(superclass_index_num_samples):
            # TESTED!
            first_non_empty_superclass = None
            second_non_empty_superclass = None
            for superclass_index, superclass_num_samples in enumerate(superclass_index_num_samples):
                if superclass_num_samples > 0:
                    if first_non_empty_superclass is None:
                        first_non_empty_superclass = superclass_index
                    else:
                        second_non_empty_superclass = superclass_index
                        break
            return first_non_empty_superclass, second_non_empty_superclass

        def _calculate_value_class_ratio(values_seen, values_num_samples,
                                         superclass_contingency_table, non_empty_class_indices):
            # TESTED!
            value_class_ratio = [] # [(value, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = superclass_contingency_table[
                    curr_value][second_class_index]
                value_class_ratio.append(
                    (curr_value, number_second_non_empty / values_num_samples[curr_value]))
            value_class_ratio.sort(key=lambda tup: tup[1])
            return value_class_ratio


        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_superclass,
         second_non_empty_superclass) = _get_non_empty_superclass_indices(
             superclass_index_num_samples)
        if first_non_empty_superclass is None or second_non_empty_superclass is None:
            return (float('+inf'), {0}, set())

        value_class_ratio = _calculate_value_class_ratio(values_seen,
                                                         values_num_samples,
                                                         superclass_contingency_table,
                                                         (first_non_empty_superclass,
                                                          second_non_empty_superclass))

        best_split_children_gini_gain = float('+inf')
        best_last_left_index = 0

        num_right_samples = num_total_valid_samples
        class_num_right = np.copy(class_index_num_samples)
        num_left_samples = 0
        class_num_left = np.zeros(class_num_right.shape, dtype=int)

        for last_left_index, (last_left_value, _) in enumerate(value_class_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            num_left_samples += num_samples_last_left_value
            num_right_samples -= num_samples_last_left_value
            class_num_left += contingency_table[last_left_value]
            class_num_right -= contingency_table[last_left_value]

            curr_children_gini_index = cls._calculate_children_gini_index(num_left_samples,
                                                                          class_num_left,
                                                                          num_right_samples,
                                                                          class_num_right)
            if curr_children_gini_index < best_split_children_gini_gain:
                best_split_children_gini_gain = curr_children_gini_index
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set(tup[0] for tup in value_class_ratio[:best_last_left_index + 1])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_children_gini_gain, set_left_values, set_right_values)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 LARGEST CLASS ALONE                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class LargestClassAlone(Criterion):
    """Largest Class Alone criterion."""
    name = 'Largest Class Alone'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the Hypercube Cover
        criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                largest_class_index, _ = max(
                    enumerate(tree_node.class_index_num_samples), key=lambda x: x[1])
                (superclass_contingency_table,
                 superclass_index_num_samples) = cls._get_superclass_contingency_table(
                     tree_node.contingency_tables[attrib_index].contingency_table,
                     tree_node.contingency_tables[attrib_index].values_num_samples,
                     tree_node.class_index_num_samples,
                     largest_class_index)
                (curr_gini_gain,
                 left_values,
                 right_values) = cls._two_class_trick(
                     tree_node.class_index_num_samples,
                     superclass_index_num_samples,
                     values_seen,
                     tree_node.contingency_tables[attrib_index].contingency_table,
                     tree_node.contingency_tables[attrib_index].values_num_samples,
                     superclass_contingency_table,
                     len(tree_node.valid_samples_indices))
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[left_values, right_values],
                          criterion_value=curr_gini_gain))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_gini,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_gini))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_superclass_contingency_table(contingency_table, values_num_samples,
                                          class_index_num_samples, largest_classes_index):
        superclass_contingency_table = np.array(
            [contingency_table[:, largest_classes_index],
             values_num_samples - contingency_table[:, largest_classes_index]
            ]).T
        superclass_index_num_samples = [
            class_index_num_samples[largest_classes_index],
            sum(class_index_num_samples) - class_index_num_samples[largest_classes_index]]
        return superclass_contingency_table, superclass_index_num_samples

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_gini = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                gini_value = cls._calculate_children_gini_index(num_left_samples,
                                                                class_num_left,
                                                                num_right_samples,
                                                                class_num_right)
                if gini_value < best_gini:
                    best_gini = gini_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_gini, best_last_left_value, best_first_right_value)

    @staticmethod
    def _calculate_gini_index(side_num, class_num_side):
        gini_index = 1.0
        for curr_class_num_side in class_num_side:
            if curr_class_num_side > 0:
                gini_index -= (curr_class_num_side / side_num) ** 2
        return gini_index

    @classmethod
    def _calculate_children_gini_index(cls, left_num, class_num_left, right_num, class_num_right):
        left_split_gini_index = cls._calculate_gini_index(left_num, class_num_left)
        right_split_gini_index = cls._calculate_gini_index(right_num, class_num_right)
        children_gini_index = ((left_num * left_split_gini_index
                                + right_num * right_split_gini_index)
                               / (left_num + right_num))
        return children_gini_index

    @classmethod
    def _two_class_trick(cls, class_index_num_samples, superclass_index_num_samples, values_seen,
                         contingency_table, values_num_samples, superclass_contingency_table,
                         num_total_valid_samples):
        # TESTED!
        def _get_non_empty_superclass_indices(superclass_index_num_samples):
            # TESTED!
            first_non_empty_superclass = None
            second_non_empty_superclass = None
            for superclass_index, superclass_num_samples in enumerate(superclass_index_num_samples):
                if superclass_num_samples > 0:
                    if first_non_empty_superclass is None:
                        first_non_empty_superclass = superclass_index
                    else:
                        second_non_empty_superclass = superclass_index
                        break
            return first_non_empty_superclass, second_non_empty_superclass

        def _calculate_value_class_ratio(values_seen, values_num_samples,
                                         superclass_contingency_table, non_empty_class_indices):
            # TESTED!
            value_class_ratio = [] # [(value, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = superclass_contingency_table[
                    curr_value][second_class_index]
                value_class_ratio.append(
                    (curr_value, number_second_non_empty / values_num_samples[curr_value]))
            value_class_ratio.sort(key=lambda tup: tup[1])
            return value_class_ratio


        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_superclass,
         second_non_empty_superclass) = _get_non_empty_superclass_indices(
             superclass_index_num_samples)
        if first_non_empty_superclass is None or second_non_empty_superclass is None:
            return (float('+inf'), {0}, set())

        value_class_ratio = _calculate_value_class_ratio(values_seen,
                                                         values_num_samples,
                                                         superclass_contingency_table,
                                                         (first_non_empty_superclass,
                                                          second_non_empty_superclass))

        best_split_children_gini_gain = float('+inf')
        best_last_left_index = 0

        num_right_samples = num_total_valid_samples
        class_num_right = np.copy(class_index_num_samples)
        num_left_samples = 0
        class_num_left = np.zeros(class_num_right.shape, dtype=int)

        for last_left_index, (last_left_value, _) in enumerate(value_class_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            num_left_samples += num_samples_last_left_value
            num_right_samples -= num_samples_last_left_value
            class_num_left += contingency_table[last_left_value]
            class_num_right -= contingency_table[last_left_value]

            curr_children_gini_index = cls._calculate_children_gini_index(num_left_samples,
                                                                          class_num_left,
                                                                          num_right_samples,
                                                                          class_num_right)
            if curr_children_gini_index < best_split_children_gini_gain:
                best_split_children_gini_gain = curr_children_gini_index
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set(tup[0] for tup in value_class_ratio[:best_last_left_index + 1])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_children_gini_gain, set_left_values, set_right_values)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                       SLIQ-Ext                                            ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class SliqExt(Criterion):
    """SLIQ-Ext criterion using the Gini impurity measure."""
    name = 'SLIQ-ext'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the SLIQ-Ext criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                (best_gini,
                 left_values,
                 right_values) = cls._get_best_attribute_split(
                     values_seen,
                     tree_node.contingency_tables[attrib_index].contingency_table,
                     tree_node.contingency_tables[attrib_index].values_num_samples)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[left_values, right_values],
                          criterion_value=best_gini))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_gini,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_gini))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @classmethod
    def _calculate_split_gini_index(cls, contingency_table, values_num_samples, left_values,
                                    right_values):
        """Calculates the weighted Gini index of a split."""
        num_left_samples, num_right_samples = cls._get_num_samples_per_side(
            values_num_samples, left_values, right_values)
        num_samples_per_class_left = cls._get_num_samples_per_class_in_values(
            contingency_table, left_values)
        num_samples_per_class_right = cls._get_num_samples_per_class_in_values(
            contingency_table, right_values)
        return cls._get_gini_value(num_samples_per_class_left, num_samples_per_class_right,
                                   num_left_samples, num_right_samples)

    @classmethod
    def _get_gini_value(cls, num_samples_per_class_left, num_samples_per_class_right,
                        num_left_samples, num_right_samples):
        """Calculates the weighted Gini index of a split."""
        num_samples = num_left_samples + num_right_samples
        left_gini = cls._calculate_node_gini_index(num_left_samples, num_samples_per_class_left)
        right_gini = cls._calculate_node_gini_index(num_right_samples, num_samples_per_class_right)
        return ((num_left_samples / num_samples) * left_gini +
                (num_right_samples / num_samples) * right_gini)

    @staticmethod
    def _calculate_node_gini_index(num_split_samples, num_samples_per_class_in_split):
        """Calculates the Gini index of a node."""
        if not num_split_samples:
            return 1.0
        gini_index = 1.0
        for curr_class_num_samples in num_samples_per_class_in_split:
            gini_index -= (curr_class_num_samples / num_split_samples)**2
        return gini_index

    @staticmethod
    def _get_num_samples_per_side(values_num_samples, left_values, right_values):
        """Returns two sets, each containing the values of a split side."""
        num_left_samples = sum(values_num_samples[value] for value in left_values)
        num_right_samples = sum(values_num_samples[value] for value in right_values)
        return  num_left_samples, num_right_samples

    @staticmethod
    def _get_num_samples_per_class_in_values(contingency_table, values):
        """Returns a list, i-th entry contains the number of samples of class i."""
        num_classes = contingency_table.shape[1]
        num_samples_per_class = [0] * num_classes
        for value in values:
            for class_index in range(num_classes):
                num_samples_per_class[class_index] += contingency_table[
                    value, class_index]
        return num_samples_per_class

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _get_best_attribute_split(cls, values_seen, contingency_table, num_samples_per_value):
        """Gets the attribute's best split according to the SLIQ-ext criterion."""
        best_gini = float('+inf')
        best_left_values = set()
        best_right_values = set()
        curr_left_values = set(values_seen)
        curr_right_values = set()
        while curr_left_values:
            iteration_best_gini = float('+inf')
            iteration_best_left_values = set()
            iteration_best_right_values = set()
            for value in curr_left_values:
                curr_left_values = curr_left_values - set([value])
                curr_right_values = curr_right_values | set([value])
                curr_gini = cls._calculate_split_gini_index(contingency_table,
                                                            num_samples_per_value,
                                                            curr_left_values,
                                                            curr_right_values)
                if curr_gini < iteration_best_gini:
                    iteration_best_gini = curr_gini
                    iteration_best_left_values = set(curr_left_values)
                    iteration_best_right_values = set(curr_right_values)
            if iteration_best_gini < best_gini:
                best_gini = iteration_best_gini
                best_left_values = set(iteration_best_left_values)
                best_right_values = set(iteration_best_right_values)
            curr_left_values = iteration_best_left_values
            curr_right_values = iteration_best_right_values
        return best_gini, best_left_values, best_right_values

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_gini = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                gini_value = cls._get_gini_value(class_num_left,
                                                 class_num_right,
                                                 num_left_samples,
                                                 num_right_samples)
                if gini_value < best_gini:
                    best_gini = gini_value
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_gini, best_last_left_value, best_first_right_value)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                     PC-ext-Entropy                                        ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class PCExtEntropy(Criterion):
    """PC-ext criterion using the Entropy impurity measure."""
    name = 'PC-ext-Entropy'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the PC-ext criterion.

        Uses the Information Gain impurity measure.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                contingency_table = tree_node.contingency_tables[attrib_index].contingency_table
                values_num_samples = tree_node.contingency_tables[
                    attrib_index].values_num_samples
                (new_contingency_table,
                 new_num_samples_per_value,
                 new_index_to_old) = cls._group_values(contingency_table, values_num_samples)
                principal_component = cls._get_principal_component(
                    len(tree_node.valid_samples_indices),
                    new_contingency_table,
                    new_num_samples_per_value)
                inner_product_results = np.dot(principal_component, new_contingency_table.T)
                new_indices_order = inner_product_results.argsort()

                best_entropy = float('+inf')
                best_left_values = set()
                best_right_values = set()
                left_values = set()
                right_values = set(new_indices_order)
                for metaindex, first_right in enumerate(new_indices_order):
                    curr_split_impurity = cls._calculate_information_gain(
                        new_contingency_table,
                        new_num_samples_per_value,
                        left_values,
                        right_values)
                    if curr_split_impurity < best_entropy:
                        best_entropy = curr_split_impurity
                        best_left_values = set(left_values)
                        best_right_values = set(right_values)
                    if left_values: # extended splits
                        last_left = new_indices_order[metaindex - 1]
                        left_values.remove(last_left)
                        right_values.add(last_left)
                        right_values.remove(first_right)
                        left_values.add(first_right)
                        curr_ext_split_impurity = cls._calculate_information_gain(
                            new_contingency_table,
                            new_num_samples_per_value,
                            left_values,
                            right_values)
                        if curr_ext_split_impurity < best_entropy:
                            best_entropy = curr_ext_split_impurity
                            best_left_values = set(left_values)
                            best_right_values = set(right_values)
                        right_values.remove(last_left)
                        left_values.add(last_left)
                        left_values.remove(first_right)
                        right_values.add(first_right)
                    right_values.remove(first_right)
                    left_values.add(first_right)
                (best_left_old_values,
                 best_right_old_values) = cls._change_split_to_use_old_values(best_left_values,
                                                                              best_right_values,
                                                                              new_index_to_old)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[best_left_old_values, best_right_old_values],
                          criterion_value=best_entropy))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_entropy,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_entropy))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @classmethod
    def _calculate_information_gain(cls, contingency_table, num_samples_per_value, left_values,
                                    right_values):
        """Calculates the Information Gain of the given binary split."""
        num_left_samples, num_right_samples = cls._get_num_samples_per_side(
            num_samples_per_value, left_values, right_values)
        num_samples_per_class_left = cls._get_num_samples_per_class_in_values(
            contingency_table, left_values)
        num_samples_per_class_right = cls._get_num_samples_per_class_in_values(
            contingency_table, right_values)
        return cls._get_information_gain_value(num_samples_per_class_left,
                                               num_samples_per_class_right,
                                               num_left_samples,
                                               num_right_samples)

    @classmethod
    def _get_information_gain_value(cls, num_samples_per_class_left, num_samples_per_class_right,
                                    num_left_samples, num_right_samples):
        """Calculates the weighted Information Gain of a split."""
        num_samples = num_left_samples + num_right_samples
        left_entropy = cls._calculate_node_information(
            num_left_samples, num_samples_per_class_left)
        right_entropy = cls._calculate_node_information(
            num_right_samples, num_samples_per_class_right)
        split_information_gain = ((num_left_samples / num_samples) * left_entropy +
                                  (num_right_samples / num_samples) * right_entropy)
        return split_information_gain

    @classmethod
    def _calculate_node_information(cls, num_split_samples, num_samples_per_class_in_split):
        """Calculates the Information of the node given by the values."""
        information = 0.0
        for curr_class_num_samples in num_samples_per_class_in_split:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_split_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_entropy = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                information_gain = cls._get_information_gain_value(class_num_left,
                                                                   class_num_right,
                                                                   num_left_samples,
                                                                   num_right_samples)
                if information_gain < best_entropy:
                    best_entropy = information_gain
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_entropy, best_last_left_value, best_first_right_value)

    @staticmethod
    def _get_num_samples_per_side(values_num_samples, left_values, right_values):
        """Returns two sets, each containing the values of a split side."""
        num_left_samples = sum(values_num_samples[value] for value in left_values)
        num_right_samples = sum(values_num_samples[value] for value in right_values)
        return  num_left_samples, num_right_samples

    @staticmethod
    def _get_num_samples_per_class_in_values(contingency_table, values):
        """Returns a list, i-th entry contains the number of samples of class i."""
        num_classes = contingency_table.shape[1]
        num_samples_per_class = [0] * num_classes
        for value in values:
            for class_index in range(num_classes):
                num_samples_per_class[class_index] += contingency_table[
                    value, class_index]
        return num_samples_per_class

    @classmethod
    def _group_values(cls, contingency_table, values_num_samples):
        """Groups values that have the same class probability vector. Remove empty values."""
        (interm_to_orig_value_int,
         interm_contingency_table,
         interm_values_num_samples) = cls._remove_empty_values(contingency_table,
                                                               values_num_samples)
        prob_matrix_transposed = np.divide(interm_contingency_table.T, interm_values_num_samples)
        prob_matrix = prob_matrix_transposed.T
        row_order = np.lexsort(prob_matrix_transposed[::-1])
        compared_index = row_order[0]
        new_index_to_old = [[interm_to_orig_value_int[compared_index]]]
        for interm_index in row_order[1:]:
            if np.allclose(prob_matrix[compared_index], prob_matrix[interm_index]):
                new_index_to_old[-1].append(interm_to_orig_value_int[interm_index])
            else:
                compared_index = interm_index
                new_index_to_old.append([interm_to_orig_value_int[compared_index]])
        new_num_values = len(new_index_to_old)
        num_classes = interm_contingency_table.shape[1]
        new_contingency_table = np.zeros((new_num_values, num_classes), dtype=int)
        new_num_samples_per_value = np.zeros((new_num_values), dtype=int)
        for new_index, old_indices in enumerate(new_index_to_old):
            new_contingency_table[new_index] = np.sum(contingency_table[old_indices, :], axis=0)
            new_num_samples_per_value[new_index] = np.sum(values_num_samples[old_indices])
        return new_contingency_table, new_num_samples_per_value, new_index_to_old

    @staticmethod
    def _remove_empty_values(contingency_table, values_num_samples):
        # Define conversion from original values to new values
        orig_to_new_value_int = {}
        new_to_orig_value_int = []
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                orig_to_new_value_int[orig_value] = len(new_to_orig_value_int)
                new_to_orig_value_int.append(orig_value)

        # Generate the new contingency tables
        new_contingency_table = np.zeros((len(new_to_orig_value_int), contingency_table.shape[1]),
                                         dtype=int)
        new_value_num_seen = np.zeros((len(new_to_orig_value_int)), dtype=int)
        for orig_value, curr_num_samples in enumerate(values_num_samples):
            if curr_num_samples > 0:
                curr_new_value = orig_to_new_value_int[orig_value]
                new_value_num_seen[curr_new_value] = curr_num_samples
                np.copyto(dst=new_contingency_table[curr_new_value, :],
                          src=contingency_table[orig_value, :])

        return (new_to_orig_value_int,
                new_contingency_table,
                new_value_num_seen)

    @staticmethod
    def _change_split_to_use_old_values(new_left, new_right, new_index_to_old):
        """Change split values to use indices of original contingency table."""
        left_old_values = set()
        for new_index in new_left:
            left_old_values |= set(new_index_to_old[new_index])
        right_old_values = set()
        for new_index in new_right:
            right_old_values |= set(new_index_to_old[new_index])
        return left_old_values, right_old_values

    @classmethod
    def _get_principal_component(cls, num_samples, contingency_table, values_num_samples):
        """Returns the principal component of the weighted covariance matrix."""
        num_samples_per_class = cls._get_num_samples_per_class(contingency_table)
        avg_prob_per_class = np.divide(num_samples_per_class, num_samples)
        prob_matrix = contingency_table / values_num_samples[:, None]
        diff_prob_matrix = (prob_matrix - avg_prob_per_class).T
        weight_diff_prob = diff_prob_matrix * values_num_samples[None, :]
        weighted_squared_diff_prob_matrix = np.dot(weight_diff_prob, diff_prob_matrix.T)
        weighted_covariance_matrix = (1/(num_samples - 1)) * weighted_squared_diff_prob_matrix
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_covariance_matrix)
        index_largest_eigenvalue = np.argmax(np.square(eigenvalues))
        return eigenvectors[:, index_largest_eigenvalue]

    @staticmethod
    def _get_num_samples_per_class(contingency_table):
        """Returns a list, i-th entry contains the number of samples of class i."""
        num_values, num_classes = contingency_table.shape
        num_samples_per_class = [0] * num_classes
        for value in range(num_values):
            for class_index in range(num_classes):
                num_samples_per_class[class_index] += contingency_table[value, class_index]
        return num_samples_per_class



#################################################################################################
#################################################################################################
###                                                                                           ###
###                                 HYPERCUBE COVER-ENTROPY                                   ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class HypercubeCoverEntropy(Criterion):
    """Hypercube Cover criterion using the Entropy impurity measure."""
    name = 'Hypercube Cover-Entropy'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the Hypercube Cover
        criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                best_entropy = float('+inf')
                best_left_values = set()
                best_right_values = set()
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                for (set_left_classes,
                     set_right_classes) in cls._generate_superclasses(
                         tree_node.class_index_num_samples):
                    (superclass_contingency_table,
                     superclass_index_num_samples) = cls._get_superclass_contingency_table(
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         set_left_classes,
                         set_right_classes)
                    (curr_entropy,
                     left_values,
                     right_values) = cls._two_class_trick(
                         tree_node.class_index_num_samples,
                         superclass_index_num_samples,
                         values_seen,
                         tree_node.contingency_tables[attrib_index].contingency_table,
                         tree_node.contingency_tables[attrib_index].values_num_samples,
                         superclass_contingency_table,
                         len(tree_node.valid_samples_indices))

                    if curr_entropy < best_entropy:
                        best_entropy = curr_entropy
                        best_left_values = left_values
                        best_right_values = right_values
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[best_left_values, best_right_values],
                          criterion_value=best_entropy))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_entropy,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_entropy))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @classmethod
    def _get_information_gain_value(cls, num_samples_per_class_left, num_samples_per_class_right,
                                    num_left_samples, num_right_samples):
        """Calculates the weighted Information Gain of a split."""
        num_samples = num_left_samples + num_right_samples
        left_entropy = cls._calculate_node_information(
            num_left_samples, num_samples_per_class_left)
        right_entropy = cls._calculate_node_information(
            num_right_samples, num_samples_per_class_right)
        split_information_gain = ((num_left_samples / num_samples) * left_entropy +
                                  (num_right_samples / num_samples) * right_entropy)
        return split_information_gain

    @classmethod
    def _calculate_node_information(cls, num_split_samples, num_samples_per_class_in_split):
        """Calculates the Information of the node given by the values."""
        information = 0.0
        for curr_class_num_samples in num_samples_per_class_in_split:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_split_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @staticmethod
    def _generate_superclasses(class_index_num_samples):
        # We only need to look at superclasses of up to (len(class_index_num_samples)/2 + 1)
        # elements because of symmetry! The subsets we are not choosing are complements of the ones
        # chosen.
        non_empty_classes = set([])
        for class_index, class_num_samples in enumerate(class_index_num_samples):
            if class_num_samples > 0:
                non_empty_classes.add(class_index)
        number_non_empty_classes = len(non_empty_classes)

        for left_classes in itertools.chain.from_iterable(
                itertools.combinations(non_empty_classes, size_left_superclass)
                for size_left_superclass in range(1, number_non_empty_classes // 2 + 1)):
            set_left_classes = set(left_classes)
            set_right_classes = non_empty_classes - set_left_classes
            if not set_left_classes or not set_right_classes:
                # A valid split must have at least one sample in each side
                continue
            yield set_left_classes, set_right_classes

    @staticmethod
    def _get_superclass_contingency_table(contingency_table, values_num_samples, set_left_classes,
                                          set_right_classes):
        superclass_contingency_table = np.zeros((contingency_table.shape[0], 2), dtype=float)
        superclass_index_num_samples = [0, 0]
        for value, value_num_samples in enumerate(values_num_samples):
            if value_num_samples == 0:
                continue
            for class_index in set_left_classes:
                superclass_index_num_samples[0] += contingency_table[value][class_index]
                superclass_contingency_table[value][0] += contingency_table[value][class_index]
            for class_index in set_right_classes:
                superclass_index_num_samples[1] += contingency_table[value][class_index]
                superclass_contingency_table[value][1] += contingency_table[value][class_index]
        return superclass_contingency_table, superclass_index_num_samples

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_entropy = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                information_gain = cls._get_information_gain_value(class_num_left,
                                                                   class_num_right,
                                                                   num_left_samples,
                                                                   num_right_samples)
                if information_gain < best_entropy:
                    best_entropy = information_gain
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_entropy, best_last_left_value, best_first_right_value)

    @classmethod
    def _two_class_trick(cls, class_index_num_samples, superclass_index_num_samples, values_seen,
                         contingency_table, values_num_samples, superclass_contingency_table,
                         num_total_valid_samples):
        # TESTED!
        def _get_non_empty_superclass_indices(superclass_index_num_samples):
            # TESTED!
            first_non_empty_superclass = None
            second_non_empty_superclass = None
            for superclass_index, superclass_num_samples in enumerate(superclass_index_num_samples):
                if superclass_num_samples > 0:
                    if first_non_empty_superclass is None:
                        first_non_empty_superclass = superclass_index
                    else:
                        second_non_empty_superclass = superclass_index
                        break
            return first_non_empty_superclass, second_non_empty_superclass

        def _calculate_value_class_ratio(values_seen, values_num_samples,
                                         superclass_contingency_table, non_empty_class_indices):
            # TESTED!
            value_class_ratio = [] # [(value, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = superclass_contingency_table[
                    curr_value][second_class_index]
                value_class_ratio.append(
                    (curr_value, number_second_non_empty / values_num_samples[curr_value]))
            value_class_ratio.sort(key=lambda tup: tup[1])
            return value_class_ratio


        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_superclass,
         second_non_empty_superclass) = _get_non_empty_superclass_indices(
             superclass_index_num_samples)
        if first_non_empty_superclass is None or second_non_empty_superclass is None:
            return (float('+inf'), {0}, set())

        value_class_ratio = _calculate_value_class_ratio(values_seen,
                                                         values_num_samples,
                                                         superclass_contingency_table,
                                                         (first_non_empty_superclass,
                                                          second_non_empty_superclass))

        best_split_entropy = float('+inf')
        best_last_left_index = 0

        num_right_samples = num_total_valid_samples
        class_num_right = np.copy(class_index_num_samples)
        num_left_samples = 0
        class_num_left = np.zeros(class_num_right.shape, dtype=int)

        for last_left_index, (last_left_value, _) in enumerate(value_class_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            num_left_samples += num_samples_last_left_value
            num_right_samples -= num_samples_last_left_value
            class_num_left += contingency_table[last_left_value]
            class_num_right -= contingency_table[last_left_value]

            curr_information_gain = cls._get_information_gain_value(class_num_left,
                                                                    class_num_right,
                                                                    num_left_samples,
                                                                    num_right_samples)
            if curr_information_gain < best_split_entropy:
                best_split_entropy = curr_information_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set(tup[0] for tup in value_class_ratio[:best_last_left_index + 1])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_entropy, set_left_values, set_right_values)



#################################################################################################
#################################################################################################
###                                                                                           ###
###                               LARGEST CLASS ALONE-ENTROPY                                 ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class LargestClassAloneEntropy(Criterion):
    """Largest Class Alone criterion using the Entropy impurity measure."""
    name = 'Largest Class Alone-Entropy'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the Hypercube Cover
        criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                largest_class_index, _ = max(
                    enumerate(tree_node.class_index_num_samples), key=lambda x: x[1])
                (superclass_contingency_table,
                 superclass_index_num_samples) = cls._get_superclass_contingency_table(
                     tree_node.contingency_tables[attrib_index].contingency_table,
                     tree_node.contingency_tables[attrib_index].values_num_samples,
                     tree_node.class_index_num_samples,
                     largest_class_index)
                (best_entropy,
                 left_values,
                 right_values) = cls._two_class_trick(
                     tree_node.class_index_num_samples,
                     superclass_index_num_samples,
                     values_seen,
                     tree_node.contingency_tables[attrib_index].contingency_table,
                     tree_node.contingency_tables[attrib_index].values_num_samples,
                     superclass_contingency_table,
                     len(tree_node.valid_samples_indices))
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[left_values, right_values],
                          criterion_value=best_entropy))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_entropy,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_entropy))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @classmethod
    def _get_information_gain_value(cls, num_samples_per_class_left, num_samples_per_class_right,
                                    num_left_samples, num_right_samples):
        """Calculates the weighted Information Gain of a split."""
        num_samples = num_left_samples + num_right_samples
        left_entropy = cls._calculate_node_information(
            num_left_samples, num_samples_per_class_left)
        right_entropy = cls._calculate_node_information(
            num_right_samples, num_samples_per_class_right)
        split_information_gain = ((num_left_samples / num_samples) * left_entropy +
                                  (num_right_samples / num_samples) * right_entropy)
        return split_information_gain

    @classmethod
    def _calculate_node_information(cls, num_split_samples, num_samples_per_class_in_split):
        """Calculates the Information of the node given by the values."""
        information = 0.0
        for curr_class_num_samples in num_samples_per_class_in_split:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_split_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_superclass_contingency_table(contingency_table, values_num_samples,
                                          class_index_num_samples, largest_classes_index):
        superclass_contingency_table = np.array(
            [contingency_table[:, largest_classes_index],
             values_num_samples - contingency_table[:, largest_classes_index]
            ]).T
        superclass_index_num_samples = [
            class_index_num_samples[largest_classes_index],
            sum(class_index_num_samples) - class_index_num_samples[largest_classes_index]]
        return superclass_contingency_table, superclass_index_num_samples

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_entropy = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                information_gain = cls._get_information_gain_value(class_num_left,
                                                                   class_num_right,
                                                                   num_left_samples,
                                                                   num_right_samples)
                if information_gain < best_entropy:
                    best_entropy = information_gain
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_entropy, best_last_left_value, best_first_right_value)

    @classmethod
    def _two_class_trick(cls, class_index_num_samples, superclass_index_num_samples, values_seen,
                         contingency_table, values_num_samples, superclass_contingency_table,
                         num_total_valid_samples):
        # TESTED!
        def _get_non_empty_superclass_indices(superclass_index_num_samples):
            # TESTED!
            first_non_empty_superclass = None
            second_non_empty_superclass = None
            for superclass_index, superclass_num_samples in enumerate(superclass_index_num_samples):
                if superclass_num_samples > 0:
                    if first_non_empty_superclass is None:
                        first_non_empty_superclass = superclass_index
                    else:
                        second_non_empty_superclass = superclass_index
                        break
            return first_non_empty_superclass, second_non_empty_superclass

        def _calculate_value_class_ratio(values_seen, values_num_samples,
                                         superclass_contingency_table, non_empty_class_indices):
            # TESTED!
            value_class_ratio = [] # [(value, ratio_on_second_class)]
            second_class_index = non_empty_class_indices[1]
            for curr_value in values_seen:
                number_second_non_empty = superclass_contingency_table[
                    curr_value][second_class_index]
                value_class_ratio.append(
                    (curr_value, number_second_non_empty / values_num_samples[curr_value]))
            value_class_ratio.sort(key=lambda tup: tup[1])
            return value_class_ratio


        # We only need to sort values by the percentage of samples in second non-empty class with
        # this value. The best split will be given by choosing an index to split this list of
        # values in two.
        (first_non_empty_superclass,
         second_non_empty_superclass) = _get_non_empty_superclass_indices(
             superclass_index_num_samples)
        if first_non_empty_superclass is None or second_non_empty_superclass is None:
            return (float('+inf'), {0}, set())

        value_class_ratio = _calculate_value_class_ratio(values_seen,
                                                         values_num_samples,
                                                         superclass_contingency_table,
                                                         (first_non_empty_superclass,
                                                          second_non_empty_superclass))

        best_split_entropy = float('+inf')
        best_last_left_index = 0

        num_right_samples = num_total_valid_samples
        class_num_right = np.copy(class_index_num_samples)
        num_left_samples = 0
        class_num_left = np.zeros(class_num_right.shape, dtype=int)

        for last_left_index, (last_left_value, _) in enumerate(value_class_ratio[:-1]):
            num_samples_last_left_value = values_num_samples[last_left_value]
            # num_samples_last_left_value > 0 always, since the values without samples were not
            # added to the values_seen when created by cls._generate_value_to_index

            num_left_samples += num_samples_last_left_value
            num_right_samples -= num_samples_last_left_value
            class_num_left += contingency_table[last_left_value]
            class_num_right -= contingency_table[last_left_value]

            curr_information_gain = cls._get_information_gain_value(class_num_left,
                                                                    class_num_right,
                                                                    num_left_samples,
                                                                    num_right_samples)
            if curr_information_gain < best_split_entropy:
                best_split_entropy = curr_information_gain
                best_last_left_index = last_left_index

        # Let's get the values and split the indices corresponding to the best split found.
        set_left_values = set(tup[0] for tup in value_class_ratio[:best_last_left_index + 1])
        set_right_values = set(values_seen) - set_left_values

        return (best_split_entropy, set_left_values, set_right_values)




#################################################################################################
#################################################################################################
###                                                                                           ###
###                                    SLIQ-Ext-ENTROPY                                       ###
###                                                                                           ###
#################################################################################################
#################################################################################################

class SliqExtEntropy(Criterion):
    """SLIQ-Ext criterion using the Entropy impurity measure."""
    name = 'SLIQ-ext-Entropy'

    @classmethod
    def select_best_attribute_and_split(cls, tree_node):
        """Returns the best attribute and its best split, according to the SLIQ-Ext criterion.

        Args:
          tree_node (TreeNode): tree node where we want to find the best attribute/split.

        Returns the best split found.
        """
        best_splits_per_attrib = []
        for (attrib_index,
             (is_valid_nominal_attrib,
              is_valid_numeric_attrib)) in enumerate(zip(tree_node.valid_nominal_attribute,
                                                         tree_node.valid_numeric_attribute)):
            if is_valid_nominal_attrib:
                values_seen = cls._get_values_seen(
                    tree_node.contingency_tables[attrib_index].values_num_samples)
                (best_entropy,
                 left_values,
                 right_values) = cls._get_best_attribute_split(
                     values_seen,
                     tree_node.contingency_tables[attrib_index].contingency_table,
                     tree_node.contingency_tables[attrib_index].values_num_samples)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[left_values, right_values],
                          criterion_value=best_entropy))
            elif is_valid_numeric_attrib:
                values_and_classes = cls._get_numeric_values_seen(tree_node.valid_samples_indices,
                                                                  tree_node.dataset.samples,
                                                                  tree_node.dataset.sample_class,
                                                                  attrib_index)
                values_and_classes.sort()
                (best_entropy,
                 last_left_value,
                 first_right_value) = cls._solve_for_numeric(
                     values_and_classes,
                     tree_node.dataset.num_classes)
                best_splits_per_attrib.append(
                    Split(attrib_index=attrib_index,
                          splits_values=[{last_left_value}, {first_right_value}],
                          criterion_value=best_entropy))
        if best_splits_per_attrib:
            return min(best_splits_per_attrib, key=lambda split: split.criterion_value)
        return Split()

    @classmethod
    def _calculate_information_gain(cls, contingency_table, num_samples_per_value, left_values,
                                    right_values):
        """Calculates the Information Gain of the given binary split."""
        num_left_samples, num_right_samples = cls._get_num_samples_per_side(
            num_samples_per_value, left_values, right_values)
        num_samples_per_class_left = cls._get_num_samples_per_class_in_values(
            contingency_table, left_values)
        num_samples_per_class_right = cls._get_num_samples_per_class_in_values(
            contingency_table, right_values)
        return cls._get_information_gain_value(num_samples_per_class_left,
                                               num_samples_per_class_right,
                                               num_left_samples,
                                               num_right_samples)

    @classmethod
    def _get_information_gain_value(cls, num_samples_per_class_left, num_samples_per_class_right,
                                    num_left_samples, num_right_samples):
        """Calculates the weighted Information Gain of a split."""
        num_samples = num_left_samples + num_right_samples
        left_entropy = cls._calculate_node_information(
            num_left_samples, num_samples_per_class_left)
        right_entropy = cls._calculate_node_information(
            num_right_samples, num_samples_per_class_right)
        split_information_gain = ((num_left_samples / num_samples) * left_entropy +
                                  (num_right_samples / num_samples) * right_entropy)
        return split_information_gain

    @classmethod
    def _calculate_node_information(cls, num_split_samples, num_samples_per_class_in_split):
        """Calculates the Information of the node given by the values."""
        information = 0.0
        for curr_class_num_samples in num_samples_per_class_in_split:
            if curr_class_num_samples != 0:
                curr_frequency = curr_class_num_samples / num_split_samples
                information -= curr_frequency * math.log2(curr_frequency)
        return information

    @staticmethod
    def _get_num_samples_per_side(values_num_samples, left_values, right_values):
        """Returns two sets, each containing the values of a split side."""
        num_left_samples = sum(values_num_samples[value] for value in left_values)
        num_right_samples = sum(values_num_samples[value] for value in right_values)
        return  num_left_samples, num_right_samples

    @staticmethod
    def _get_num_samples_per_class_in_values(contingency_table, values):
        """Returns a list, i-th entry contains the number of samples of class i."""
        num_classes = contingency_table.shape[1]
        num_samples_per_class = [0] * num_classes
        for value in values:
            for class_index in range(num_classes):
                num_samples_per_class[class_index] += contingency_table[
                    value, class_index]
        return num_samples_per_class

    @staticmethod
    def _get_values_seen(values_num_samples):
        values_seen = set()
        for value, num_samples in enumerate(values_num_samples):
            if num_samples > 0:
                values_seen.add(value)
        return values_seen

    @staticmethod
    def _get_numeric_values_seen(valid_samples_indices, sample, sample_class, attrib_index):
        values_and_classes = []
        for sample_index in valid_samples_indices:
            sample_value = sample[sample_index][attrib_index]
            values_and_classes.append((sample_value, sample_class[sample_index]))
        return values_and_classes

    @classmethod
    def _get_best_attribute_split(cls, values_seen, contingency_table, num_samples_per_value):
        """Gets the attribute's best split according to the SLIQ-ext criterion."""
        best_entropy = float('+inf')
        best_left_values = set()
        best_right_values = set()
        curr_left_values = set(values_seen)
        curr_right_values = set()
        while curr_left_values:
            iteration_best_entropy = float('+inf')
            iteration_best_left_values = set()
            iteration_best_right_values = set()
            for value in curr_left_values:
                curr_left_values = curr_left_values - set([value])
                curr_right_values = curr_right_values | set([value])
                curr_entropy = cls._calculate_information_gain(contingency_table,
                                                               num_samples_per_value,
                                                               curr_left_values,
                                                               curr_right_values)
                if curr_entropy < iteration_best_entropy:
                    iteration_best_entropy = curr_entropy
                    iteration_best_left_values = set(curr_left_values)
                    iteration_best_right_values = set(curr_right_values)
            if iteration_best_entropy < best_entropy:
                best_entropy = iteration_best_entropy
                best_left_values = set(iteration_best_left_values)
                best_right_values = set(iteration_best_right_values)
            curr_left_values = iteration_best_left_values
            curr_right_values = iteration_best_right_values
        return best_entropy, best_left_values, best_right_values

    @classmethod
    def _solve_for_numeric(cls, sorted_values_and_classes, num_classes):
        last_left_value = sorted_values_and_classes[0][0]
        num_left_samples = 1
        num_right_samples = len(sorted_values_and_classes) - 1

        class_num_left = [0] * num_classes
        class_num_left[sorted_values_and_classes[0][1]] = 1

        class_num_right = [0] * num_classes
        for _, sample_class in sorted_values_and_classes[1:]:
            class_num_right[sample_class] += 1

        best_entropy = float('+inf')
        best_last_left_value = None
        best_first_right_value = None

        for first_right_index in range(1, len(sorted_values_and_classes)):
            first_right_value = sorted_values_and_classes[first_right_index][0]
            if first_right_value != last_left_value:
                information_gain = cls._get_information_gain_value(class_num_left,
                                                                   class_num_right,
                                                                   num_left_samples,
                                                                   num_right_samples)
                if information_gain < best_entropy:
                    best_entropy = information_gain
                    best_last_left_value = last_left_value
                    best_first_right_value = first_right_value

                last_left_value = first_right_value

            num_left_samples += 1
            num_right_samples -= 1
            first_right_class = sorted_values_and_classes[first_right_index][1]
            class_num_left[first_right_class] += 1
            class_num_right[first_right_class] -= 1
        return (best_entropy, best_last_left_value, best_first_right_value)
