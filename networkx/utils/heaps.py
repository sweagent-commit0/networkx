"""
Min-heaps.
"""
from heapq import heappop, heappush
from itertools import count
import networkx as nx
__all__ = ['MinHeap', 'PairingHeap', 'BinaryHeap']

class MinHeap:
    """Base class for min-heaps.

    A MinHeap stores a collection of key-value pairs ordered by their values.
    It supports querying the minimum pair, inserting a new pair, decreasing the
    value in an existing pair and deleting the minimum pair.
    """

    class _Item:
        """Used by subclassess to represent a key-value pair."""
        __slots__ = ('key', 'value')

        def __init__(self, key, value):
            self.key = key
            self.value = value

        def __repr__(self):
            return repr((self.key, self.value))

    def __init__(self):
        """Initialize a new min-heap."""
        self._dict = {}

    def min(self):
        """Query the minimum key-value pair.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        pass

    def pop(self):
        """Delete the minimum pair in the heap.

        Returns
        -------
        key, value : tuple
            The key-value pair with the minimum value in the heap.

        Raises
        ------
        NetworkXError
            If the heap is empty.
        """
        pass

    def get(self, key, default=None):
        """Returns the value associated with a key.

        Parameters
        ----------
        key : hashable object
            The key to be looked up.

        default : object
            Default value to return if the key is not present in the heap.
            Default value: None.

        Returns
        -------
        value : object.
            The value associated with the key.
        """
        pass

    def insert(self, key, value, allow_increase=False):
        """Insert a new key-value pair or modify the value in an existing
        pair.

        Parameters
        ----------
        key : hashable object
            The key.

        value : object comparable with existing values.
            The value.

        allow_increase : bool
            Whether the value is allowed to increase. If False, attempts to
            increase an existing value have no effect. Default value: False.

        Returns
        -------
        decreased : bool
            True if a pair is inserted or the existing value is decreased.
        """
        pass

    def __nonzero__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __bool__(self):
        """Returns whether the heap if empty."""
        return bool(self._dict)

    def __len__(self):
        """Returns the number of key-value pairs in the heap."""
        return len(self._dict)

    def __contains__(self, key):
        """Returns whether a key exists in the heap.

        Parameters
        ----------
        key : any hashable object.
            The key to be looked up.
        """
        return key in self._dict

class PairingHeap(MinHeap):
    """A pairing heap."""

    class _Node(MinHeap._Item):
        """A node in a pairing heap.

        A tree in a pairing heap is stored using the left-child, right-sibling
        representation.
        """
        __slots__ = ('left', 'next', 'prev', 'parent')

        def __init__(self, key, value):
            super().__init__(key, value)
            self.left = None
            self.next = None
            self.prev = None
            self.parent = None

    def __init__(self):
        """Initialize a pairing heap."""
        super().__init__()
        self._root = None

    def _link(self, root, other):
        """Link two nodes, making the one with the smaller value the parent of
        the other.
        """
        pass

    def _merge_children(self, root):
        """Merge the subtrees of the root using the standard two-pass method.
        The resulting subtree is detached from the root.
        """
        pass

    def _cut(self, node):
        """Cut a node from its parent."""
        pass

class BinaryHeap(MinHeap):
    """A binary heap."""

    def __init__(self):
        """Initialize a binary heap."""
        super().__init__()
        self._heap = []
        self._count = count()