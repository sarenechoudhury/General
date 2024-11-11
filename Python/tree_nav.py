class Doll:
    """
    A class for representing Matryoshka dolls (also known as Russian
    nesting dolls) where the values are integers.  Use Doll(None, None)
    to create an empty doll.
    """

    def __init__(self, val, doll_to_nest):
        """
        Constructor for Doll

        Inputs:
            val [str]: a value
            doll_to_nest [Doll]: an instance of the Doll class
        """
        assert val is not None or \
            doll_to_nest is None

        self.val = val
        self.nested_doll = doll_to_nest


    def is_empty(self):
        """ Is the doll empty """
        return self.val is None


    def __len__(self):
        """ Computes the length of the Doll """
        if self.is_empty():
            return 0

        return 1 + len(self.nested_doll)


    def __eq__(self, other):
        """
        Determine whether self and other represent the same doll.

        Inputs:
            self [Doll]: one M-doll
            other [Doll]: another M-doll

        Returns [bool]: True, if the two dolls are equal and False
          otherwise
        """
        raise NotImplementedError

    def __repr__(self):
        """ Create a developer-facing string representation of a Doll """

        if self.is_empty():
            return "Doll(None, None)"

        return f"Doll({self.val}, {repr(self.nested_doll)})"


    def diff_count(self, other):
        """
        Determine the number of spots at which two Dolls (self and
        other) differ.

        Inputs:
            self [Doll]: one M-doll
            other [Doll]: another M-doll

        Returns [int]: the number of spots at which the dolls differ.

        """
        if self.is_empty() and other.is_empty():
            return 0
        if self.is_empty() or other.is_empty():
            return len(other) if self.is_empty() else len(self)
        if self.val != other.val:
            return 1 + self.nested_doll.diff_count(other.nested_doll)
        return self.nested_doll.diff_count(other.nested_doll)



def mk_doll_from_lst(vals):
    """
    Make a doll from a list of values

    Inputs:
        vals [List[str]]: the values for the doll.  Outermost
           value first.
    """
    doll = Doll(None, None)
    # build the doll from back to front
    for v in vals[::-1]:
        doll = Doll(v, doll)
    return doll


def find_path(p1, p2):
    """
    Determine whether it is possible traverse from p1 to p2 by
    repeatedly taking positive steps in the x direction and/or
    positive steps in the y direction. (One step at a time.)

    Inputs:
        p1 [Tuple[int, int]]: the starting point in cartesian coordinates
        p2 [Tuple[int, int]]: the target point in cartesian coordinates.

    Returns [List[Tuple[int, int]]]: a path from p1 to p2 if one exists,
      None otherwise.
    """
    x1, y1 = p1
    x2, y2 = p2
    if p1 == p2:
        return [p1]
    if x1 < x2:
        path_op1 = find_path((x1 + 1, y1), p2)
        if path_op1:
            return [p1] + path_op1

    if y1 < y2:
        path_op2 = find_path((x1, y1 + 1), p2)
        if path_op2:
            return [p1] + path_op2
    return None


def neg_leaves(t):
    """
    Count the number of leaves with negative values in the tree

    Inputs:
        t [Tree]: a non-emty tree

    Returns [int]: the number of leaves with negative values.
    """
    if not t.children:
        return 1 if t.val < 0 else 0
    return sum(neg_leaves(child) for child in t.children)

def val_counts(N, t):
    """
    Given a tree where all the values are between 0 and N (inclusive),
    return a list where the ith element contains the number of times
    i occurs in the tree.

    Inputs:
        N [int]: a non-negative integer
        t [Tree]: a tree.

    Returns [List[int]]: a list where the ith value is a count of
      the number of times the value i occurred in the tree.
    """
    counts = [0] * (N + 1)

    def count_values(node):
        """
        Helper function to recursively count the occurrences of values
        in the tree.

        Inputs:
            node (Tree): The current node being processed.

        Returns: None
        """
        counts[node.val] += 1
        for child in node.children:
            count_values(child)

    count_values(t)

    return counts


def count_less_than_paths(t, target):
    """
    Count the number of root-to-leaf paths where weight of a
    path is strictly less than the specified target weight.

    Inputs:
        t [Tree]: a tree with non-negative values
        target [int]: a non-negative target weight

    Returns [int]: the number of root-to-leaf paths with weights
      strictly less than the target weight.

    """
    assert target >= 0
    def count_paths_helper(node, current_weight):
        """
        Helper function to recursively count root-to-leaf paths
        with weights less than the target weight.

        Inputs:
            node (Tree): The current node being processed.
            current_weight (int): The current path weight.

        Returns:
            int: Number of root-to-leaf paths with weights
            less than the target weight.
        """
        if not node:
            return 0

        current_weight += node.val

        if not node.children:
            return 1 if current_weight < target else 0

        count = 0
        for child in node.children:
            count += count_paths_helper(child, current_weight)
        return count

    return count_paths_helper(t, 0)

def find_over_the_top_nodes(t, target):
    """
    Construct a list of the node identifiers for the over-the-top nodes
    in the tree.

    Inputs:
        t [Tree]: a tree with non-negative values
        target [int]: a non-negative target weight

    Returns [List[int]]: a list of the node identifiers for the
    over-the-top nodes in the tree.
    """
    assert target >= 0
    over_the_top_nodes = []

    def check_over_the_top(node, parent_weight, current_weight):
        """
        Helper function to recursively check for over-the-top nodes.

        Inputs:
            node (Tree): The current node being processed.
            parent_weight (int): The weight of the path from
            the root to the parent node.
            current_weight (int): The current path weight.

        Returns:
            None
        """
        if not node:
            return

        current_weight += node.val

        if current_weight >= target:
            if parent_weight is None or parent_weight < target:
                over_the_top_nodes.append(node.node_id)

        for child in node.children:
            check_over_the_top(child, current_weight, current_weight)

    check_over_the_top(t, None, 0)

    return over_the_top_nodes

def find_less_than_paths(t, target):
    """
    Compute a list of the root-to-leaf paths with a weight
    strictly less than the specified target weight

    Inputs:
        t [Tree]: a tree with non-negative values
        target [int]: a non-negative target weight

    Returns [List[List[int]]]: a list of paths with weights strictly
      less than the target, where a path is represented by a list of
      node identifiers.
    """
    assert target >= 0

    def find_paths_helper(node, current_path, current_weight, result):
        """
        Helper function to recursively find root-to-leaf paths
        with weight less than the target weight.

        Inputs:
            node (Tree): The current node being processed.
            current_path (list): The current path from the root to
            the current node.
            current_weight (int): The weight of the current path.
            result (list): The list to store valid root-to-leaf paths.

        Returns:
            None
        """
        if not node:
            return

        current_path.append(node.node_id)
        current_weight += node.val

        if not node.children:
            if current_weight < target:
                result.append(current_path.copy())

        for child in node.children:
            find_paths_helper(child, current_path, current_weight, result)

        current_path.pop()

    result = []

    find_paths_helper(t, [], 0, result)

    return result
