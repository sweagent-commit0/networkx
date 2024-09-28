"""
**********
Matplotlib
**********

Draw networks with matplotlib.

Examples
--------
>>> G = nx.complete_graph(5)
>>> nx.draw(G)

See Also
--------
 - :doc:`matplotlib <matplotlib:index>`
 - :func:`matplotlib.pyplot.scatter`
 - :obj:`matplotlib.patches.FancyArrowPatch`
"""
import collections
import itertools
from numbers import Number
import networkx as nx
from networkx.drawing.layout import circular_layout, kamada_kawai_layout, planar_layout, random_layout, shell_layout, spectral_layout, spring_layout
__all__ = ['draw', 'draw_networkx', 'draw_networkx_nodes', 'draw_networkx_edges', 'draw_networkx_labels', 'draw_networkx_edge_labels', 'draw_circular', 'draw_kamada_kawai', 'draw_random', 'draw_spectral', 'draw_spring', 'draw_planar', 'draw_shell']

def draw(G, pos=None, ax=None, **kwds):
    """Draw the graph G with Matplotlib.

    Draw the graph as a simple representation with no node
    labels or edge labels and using the full Matplotlib figure area
    and no axis labels by default.  See draw_networkx() for more
    full-featured drawing that allows title, axis labels etc.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :py:mod:`networkx.drawing.layout` for functions that
        compute node positions.

    ax : Matplotlib Axes object, optional
        Draw the graph in specified Matplotlib axes.

    kwds : optional keywords
        See networkx.draw_networkx() for a description of optional keywords.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    See Also
    --------
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels

    Notes
    -----
    This function has the same name as pylab.draw and pyplot.draw
    so beware when using `from networkx import *`

    since you might overwrite the pylab.draw function.

    With pyplot use

    >>> import matplotlib.pyplot as plt
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)  # networkx draw()
    >>> plt.draw()  # pyplot draw()

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html
    """
    pass

def draw_networkx(G, pos=None, arrows=None, with_labels=True, **kwds):
    """Draw the graph G using Matplotlib.

    Draw the graph with Matplotlib with options for node positions,
    labeling, titles, and many other drawing features.
    See draw() for simple drawing without labels or axes.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary, optional
        A dictionary with nodes as keys and positions as values.
        If not specified a spring layout positioning will be computed.
        See :py:mod:`networkx.drawing.layout` for functions that
        compute node positions.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).
        For directed graphs, if True draw arrowheads.
        Note: Arrows will be the same color as edges.

    arrowstyle : str (default='-\\|>' for directed graphs)
        For directed graphs, choose the style of the arrowsheads.
        For undirected graphs default to '-'

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int or list (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. A list of values can be passed in to assign a different size for arrow head's length and width.
        See `matplotlib.patches.FancyArrowPatch` for attribute `mutation_scale`
        for more info.

    with_labels :  bool (default=True)
        Set to True to draw labels on the nodes.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default=list(G))
        Draw only specified nodes

    edgelist : list (default=list(G.edges()))
        Draw only specified edges

    node_size : scalar or array (default=300)
        Size of nodes.  If an array is specified it must be the
        same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or None (default=None)
        The node and edge transparency

    cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of nodes

    vmin,vmax : float, optional
        Minimum and maximum for node colormap scaling

    linewidths : scalar or sequence (default=1.0)
        Line width of symbol border

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    style : string (default=solid line)
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    labels : dictionary (default=None)
        Node labels in a dictionary of text labels keyed by node

    font_size : int (default=12 for nodes, 10 for edges)
        Font size for text labels

    font_color : color (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    label : string, optional
        Label for graph legend

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    kwds : optional keywords
        See networkx.draw_networkx_nodes(), networkx.draw_networkx_edges(), and
        networkx.draw_networkx_labels() for a description of optional keywords.

    Notes
    -----
    For directed graphs, arrows  are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nx.draw(G)
    >>> nx.draw(G, pos=nx.spring_layout(G))  # use spring layout

    >>> import matplotlib.pyplot as plt
    >>> limits = plt.axis("off")  # turn off axis

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels
    """
    pass

def draw_networkx_nodes(G, pos, nodelist=None, node_size=300, node_color='#1f78b4', node_shape='o', alpha=None, cmap=None, vmin=None, vmax=None, ax=None, linewidths=None, edgecolors=None, label=None, margins=None, hide_ticks=True):
    """Draw the nodes of the graph G.

    This draws only the nodes of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    nodelist : list (default list(G))
        Draw only specified nodes

    node_size : scalar or array (default=300)
        Size of nodes.  If an array it must be the same length as nodelist.

    node_color : color or array of colors (default='#1f78b4')
        Node color. Can be a single color or a sequence of colors with the same
        length as nodelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the cmap and vmin,vmax parameters. See
        matplotlib.scatter for more details.

    node_shape :  string (default='o')
        The shape of the node.  Specification is as matplotlib.scatter
        marker, one of 'so^>v<dph8'.

    alpha : float or array of floats (default=None)
        The node transparency.  This can be a single alpha value,
        in which case it will be applied to all the nodes of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    cmap : Matplotlib colormap (default=None)
        Colormap for mapping intensities of nodes

    vmin,vmax : floats or None (default=None)
        Minimum and maximum for node colormap scaling

    linewidths : [None | scalar | sequence] (default=1.0)
        Line width of symbol border

    edgecolors : [None | scalar | sequence] (default = node_color)
        Colors of node borders. Can be a single color or a sequence of colors with the
        same length as nodelist. Color can be string or rgb (or rgba) tuple of floats
        from 0-1. If numeric values are specified they will be mapped to colors
        using the cmap and vmin,vmax parameters. See `~matplotlib.pyplot.scatter` for more details.

    label : [None | string]
        Label for legend

    margins : float or 2-tuple, optional
        Sets the padding for axis autoscaling. Increase margin to prevent
        clipping for nodes that are near the edges of an image. Values should
        be in the range ``[0, 1]``. See :meth:`matplotlib.axes.Axes.margins`
        for details. The default is `None`, which uses the Matplotlib default.

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    matplotlib.collections.PathCollection
        `PathCollection` of the nodes.

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> nodes = nx.draw_networkx_nodes(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_edges
    draw_networkx_labels
    draw_networkx_edge_labels
    """
    pass

class FancyArrowFactory:
    """Draw arrows with `matplotlib.patches.FancyarrowPatch`"""

    class ConnectionStyleFactory:

        def __init__(self, connectionstyles, selfloop_height, ax=None):
            import matplotlib as mpl
            import matplotlib.path
            import numpy as np
            self.ax = ax
            self.mpl = mpl
            self.np = np
            self.base_connection_styles = [mpl.patches.ConnectionStyle(cs) for cs in connectionstyles]
            self.n = len(self.base_connection_styles)
            self.selfloop_height = selfloop_height

    def __init__(self, edge_pos, edgelist, nodelist, edge_indices, node_size, selfloop_height, connectionstyle='arc3', node_shape='o', arrowstyle='-', arrowsize=10, edge_color='k', alpha=None, linewidth=1.0, style='solid', min_source_margin=0, min_target_margin=0, ax=None):
        import matplotlib as mpl
        import matplotlib.patches
        import matplotlib.pyplot as plt
        import numpy as np
        if isinstance(connectionstyle, str):
            connectionstyle = [connectionstyle]
        elif np.iterable(connectionstyle):
            connectionstyle = list(connectionstyle)
        else:
            msg = 'ConnectionStyleFactory arg `connectionstyle` must be str or iterable'
            raise nx.NetworkXError(msg)
        self.ax = ax
        self.mpl = mpl
        self.np = np
        self.edge_pos = edge_pos
        self.edgelist = edgelist
        self.nodelist = nodelist
        self.node_shape = node_shape
        self.min_source_margin = min_source_margin
        self.min_target_margin = min_target_margin
        self.edge_indices = edge_indices
        self.node_size = node_size
        self.connectionstyle_factory = self.ConnectionStyleFactory(connectionstyle, selfloop_height, ax)
        self.arrowstyle = arrowstyle
        self.arrowsize = arrowsize
        self.arrow_colors = mpl.colors.colorConverter.to_rgba_array(edge_color, alpha)
        self.linewidth = linewidth
        self.style = style
        if isinstance(arrowsize, list) and len(arrowsize) != len(edge_pos):
            raise ValueError('arrowsize should have the same length as edgelist')

    def __call__(self, i):
        (x1, y1), (x2, y2) = self.edge_pos[i]
        shrink_source = 0
        shrink_target = 0
        if self.np.iterable(self.node_size):
            source, target = self.edgelist[i][:2]
            source_node_size = self.node_size[self.nodelist.index(source)]
            target_node_size = self.node_size[self.nodelist.index(target)]
            shrink_source = self.to_marker_edge(source_node_size, self.node_shape)
            shrink_target = self.to_marker_edge(target_node_size, self.node_shape)
        else:
            shrink_source = self.to_marker_edge(self.node_size, self.node_shape)
            shrink_target = shrink_source
        shrink_source = max(shrink_source, self.min_source_margin)
        shrink_target = max(shrink_target, self.min_target_margin)
        if isinstance(self.arrowsize, list):
            mutation_scale = self.arrowsize[i]
        else:
            mutation_scale = self.arrowsize
        if len(self.arrow_colors) > i:
            arrow_color = self.arrow_colors[i]
        elif len(self.arrow_colors) == 1:
            arrow_color = self.arrow_colors[0]
        else:
            arrow_color = self.arrow_colors[i % len(self.arrow_colors)]
        if self.np.iterable(self.linewidth):
            if len(self.linewidth) > i:
                linewidth = self.linewidth[i]
            else:
                linewidth = self.linewidth[i % len(self.linewidth)]
        else:
            linewidth = self.linewidth
        if self.np.iterable(self.style) and (not isinstance(self.style, str)) and (not isinstance(self.style, tuple)):
            if len(self.style) > i:
                linestyle = self.style[i]
            else:
                linestyle = self.style[i % len(self.style)]
        else:
            linestyle = self.style
        if x1 == x2 and y1 == y2:
            connectionstyle = self.connectionstyle_factory.self_loop(self.edge_indices[i])
        else:
            connectionstyle = self.connectionstyle_factory.curved(self.edge_indices[i])
        return self.mpl.patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=self.arrowstyle, shrinkA=shrink_source, shrinkB=shrink_target, mutation_scale=mutation_scale, color=arrow_color, linewidth=linewidth, connectionstyle=connectionstyle, linestyle=linestyle, zorder=1)

def draw_networkx_edges(G, pos, edgelist=None, width=1.0, edge_color='k', style='solid', alpha=None, arrowstyle=None, arrowsize=10, edge_cmap=None, edge_vmin=None, edge_vmax=None, ax=None, arrows=None, label=None, node_size=300, nodelist=None, node_shape='o', connectionstyle='arc3', min_source_margin=0, min_target_margin=0, hide_ticks=True):
    """Draw the edges of the graph G.

    This draws only the edges of the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edgelist : collection of edge tuples (default=G.edges())
        Draw only specified edges

    width : float or array of floats (default=1.0)
        Line width of edges

    edge_color : color or array of colors (default='k')
        Edge color. Can be a single color or a sequence of colors with the same
        length as edgelist. Color can be string or rgb (or rgba) tuple of
        floats from 0-1. If numeric values are specified they will be
        mapped to colors using the edge_cmap and edge_vmin,edge_vmax parameters.

    style : string or array of strings (default='solid')
        Edge line style e.g.: '-', '--', '-.', ':'
        or words like 'solid' or 'dashed'.
        Can be a single style or a sequence of styles with the same
        length as the edge list.
        If less styles than edges are given the styles will cycle.
        If more styles than edges are given the styles will be used sequentially
        and not be exhausted.
        Also, `(offset, onoffseq)` tuples can be used as style instead of a strings.
        (See `matplotlib.patches.FancyArrowPatch`: `linestyle`)

    alpha : float or array of floats (default=None)
        The edge transparency.  This can be a single alpha value,
        in which case it will be applied to all specified edges. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    edge_cmap : Matplotlib colormap, optional
        Colormap for mapping intensities of edges

    edge_vmin,edge_vmax : floats, optional
        Minimum and maximum for edge colormap scaling

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    arrows : bool or None, optional (default=None)
        If `None`, directed graphs draw arrowheads with
        `~matplotlib.patches.FancyArrowPatch`, while undirected graphs draw edges
        via `~matplotlib.collections.LineCollection` for speed.
        If `True`, draw arrowheads with FancyArrowPatches (bendable and stylish).
        If `False`, draw edges using LineCollection (linear and fast).

        Note: Arrowheads will be the same color as edges.

    arrowstyle : str (default='-\\|>' for directed graphs)
        For directed graphs and `arrows==True` defaults to '-\\|>',
        For undirected graphs default to '-'.

        See `matplotlib.patches.ArrowStyle` for more options.

    arrowsize : int (default=10)
        For directed graphs, choose the size of the arrow head's length and
        width. See `matplotlib.patches.FancyArrowPatch` for attribute
        `mutation_scale` for more info.

    connectionstyle : string or iterable of strings (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.
        If Iterable, index indicates i'th edge key of MultiGraph

    node_size : scalar or array (default=300)
        Size of nodes. Though the nodes are not drawn with this function, the
        node size is used in determining edge positioning.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    node_shape :  string (default='o')
        The marker used for nodes, used in determining edge positioning.
        Specification is as a `matplotlib.markers` marker, e.g. one of 'so^>v<dph8'.

    label : None or string
        Label for legend

    min_source_margin : int (default=0)
        The minimum margin (gap) at the beginning of the edge at the source.

    min_target_margin : int (default=0)
        The minimum margin (gap) at the end of the edge at the target.

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
     matplotlib.collections.LineCollection or a list of matplotlib.patches.FancyArrowPatch
        If ``arrows=True``, a list of FancyArrowPatches is returned.
        If ``arrows=False``, a LineCollection is returned.
        If ``arrows=None`` (the default), then a LineCollection is returned if
        `G` is undirected, otherwise returns a list of FancyArrowPatches.

    Notes
    -----
    For directed graphs, arrows are drawn at the head end.  Arrows can be
    turned off with keyword arrows=False or by passing an arrowstyle without
    an arrow on the end.

    Be sure to include `node_size` as a keyword argument; arrows are
    drawn considering the size of nodes.

    Self-loops are always drawn with `~matplotlib.patches.FancyArrowPatch`
    regardless of the value of `arrows` or whether `G` is directed.
    When ``arrows=False`` or ``arrows=None`` and `G` is undirected, the
    FancyArrowPatches corresponding to the self-loops are not explicitly
    returned. They should instead be accessed via the ``Axes.patches``
    attribute (see examples).

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edges = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))

    >>> G = nx.DiGraph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 3)])
    >>> arcs = nx.draw_networkx_edges(G, pos=nx.spring_layout(G))
    >>> alphas = [0.3, 0.4, 0.5]
    >>> for i, arc in enumerate(arcs):  # change alpha values of arcs
    ...     arc.set_alpha(alphas[i])

    The FancyArrowPatches corresponding to self-loops are not always
    returned, but can always be accessed via the ``patches`` attribute of the
    `matplotlib.Axes` object.

    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> G = nx.Graph([(0, 1), (0, 0)])  # Self-loop at node 0
    >>> edge_collection = nx.draw_networkx_edges(G, pos=nx.circular_layout(G), ax=ax)
    >>> self_loop_fap = ax.patches[0]

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_labels
    draw_networkx_edge_labels

    """
    pass

def draw_networkx_labels(G, pos, labels=None, font_size=12, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, clip_on=True, hide_ticks=True):
    """Draw node labels on the graph G.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    labels : dictionary (default={n: n for n in G})
        Node labels in a dictionary of text labels keyed by node.
        Node-keys in labels should appear as keys in `pos`.
        If needed use: `{n:lab for n,lab in labels.items() if n in pos}`

    font_size : int (default=12)
        Font size for text labels

    font_color : color (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, (default is Matplotlib's ax.text default)
        Specify text box properties (e.g. shape, color etc.) for node labels.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    clip_on : bool (default=True)
        Turn on clipping of node labels at axis boundaries

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    dict
        `dict` of labels keyed on the nodes

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> labels = nx.draw_networkx_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_edge_labels
    """
    pass

def draw_networkx_edge_labels(G, pos, edge_labels=None, label_pos=0.5, font_size=10, font_color='k', font_family='sans-serif', font_weight='normal', alpha=None, bbox=None, horizontalalignment='center', verticalalignment='center', ax=None, rotate=True, clip_on=True, node_size=300, nodelist=None, connectionstyle='arc3', hide_ticks=True):
    """Draw edge labels.

    Parameters
    ----------
    G : graph
        A networkx graph

    pos : dictionary
        A dictionary with nodes as keys and positions as values.
        Positions should be sequences of length 2.

    edge_labels : dictionary (default=None)
        Edge labels in a dictionary of labels keyed by edge two-tuple.
        Only labels for the keys in the dictionary are drawn.

    label_pos : float (default=0.5)
        Position of edge label along edge (0=head, 0.5=center, 1=tail)

    font_size : int (default=10)
        Font size for text labels

    font_color : color (default='k' black)
        Font color string. Color can be string or rgb (or rgba) tuple of
        floats from 0-1.

    font_weight : string (default='normal')
        Font weight

    font_family : string (default='sans-serif')
        Font family

    alpha : float or None (default=None)
        The text transparency

    bbox : Matplotlib bbox, optional
        Specify text box properties (e.g. shape, color etc.) for edge labels.
        Default is {boxstyle='round', ec=(1.0, 1.0, 1.0), fc=(1.0, 1.0, 1.0)}.

    horizontalalignment : string (default='center')
        Horizontal alignment {'center', 'right', 'left'}

    verticalalignment : string (default='center')
        Vertical alignment {'center', 'top', 'bottom', 'baseline', 'center_baseline'}

    ax : Matplotlib Axes object, optional
        Draw the graph in the specified Matplotlib axes.

    rotate : bool (default=True)
        Rotate edge labels to lie parallel to edges

    clip_on : bool (default=True)
        Turn on clipping of edge labels at axis boundaries

    node_size : scalar or array (default=300)
        Size of nodes.  If an array it must be the same length as nodelist.

    nodelist : list, optional (default=G.nodes())
       This provides the node order for the `node_size` array (if it is an array).

    connectionstyle : string or iterable of strings (default="arc3")
        Pass the connectionstyle parameter to create curved arc of rounding
        radius rad. For example, connectionstyle='arc3,rad=0.2'.
        See `matplotlib.patches.ConnectionStyle` and
        `matplotlib.patches.FancyArrowPatch` for more info.
        If Iterable, index indicates i'th edge key of MultiGraph

    hide_ticks : bool, optional
        Hide ticks of axes. When `True` (the default), ticks and ticklabels
        are removed from the axes. To set ticks and tick labels to the pyplot default,
        use ``hide_ticks=False``.

    Returns
    -------
    dict
        `dict` of labels keyed by edge

    Examples
    --------
    >>> G = nx.dodecahedral_graph()
    >>> edge_labels = nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G))

    Also see the NetworkX drawing examples at
    https://networkx.org/documentation/latest/auto_examples/index.html

    See Also
    --------
    draw
    draw_networkx
    draw_networkx_nodes
    draw_networkx_edges
    draw_networkx_labels
    """
    pass

def draw_circular(G, **kwargs):
    """Draw the graph `G` with a circular layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.circular_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called. For
    repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.circular_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.circular_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_circular(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.circular_layout`
    """
    pass

def draw_kamada_kawai(G, **kwargs):
    """Draw the graph `G` with a Kamada-Kawai force-directed layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.kamada_kawai_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.kamada_kawai_layout` directly and reuse the
    result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.kamada_kawai_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_kamada_kawai(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.kamada_kawai_layout`
    """
    pass

def draw_random(G, **kwargs):
    """Draw the graph `G` with a random layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.random_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.random_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.random_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.lollipop_graph(4, 3)
    >>> nx.draw_random(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.random_layout`
    """
    pass

def draw_spectral(G, **kwargs):
    """Draw the graph `G` with a spectral 2D layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.spectral_layout(G), **kwargs)

    For more information about how node positions are determined, see
    `~networkx.drawing.layout.spectral_layout`.

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.spectral_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.spectral_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> nx.draw_spectral(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.spectral_layout`
    """
    pass

def draw_spring(G, **kwargs):
    """Draw the graph `G` with a spring layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.spring_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    `~networkx.drawing.layout.spring_layout` is also the default layout for
    `draw`, so this function is equivalent to `draw`.

    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.spring_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.spring_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(20)
    >>> nx.draw_spring(G)

    See Also
    --------
    draw
    :func:`~networkx.drawing.layout.spring_layout`
    """
    pass

def draw_shell(G, nlist=None, **kwargs):
    """Draw networkx graph `G` with shell layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.shell_layout(G, nlist=nlist), **kwargs)

    Parameters
    ----------
    G : graph
        A networkx graph

    nlist : list of list of nodes, optional
        A list containing lists of nodes representing the shells.
        Default is `None`, meaning all nodes are in a single shell.
        See `~networkx.drawing.layout.shell_layout` for details.

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.shell_layout` directly and reuse the result::

        >>> G = nx.complete_graph(5)
        >>> pos = nx.shell_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> shells = [[0], [1, 2, 3]]
    >>> nx.draw_shell(G, nlist=shells)

    See Also
    --------
    :func:`~networkx.drawing.layout.shell_layout`
    """
    pass

def draw_planar(G, **kwargs):
    """Draw a planar networkx graph `G` with planar layout.

    This is a convenience function equivalent to::

        nx.draw(G, pos=nx.planar_layout(G), **kwargs)

    Parameters
    ----------
    G : graph
        A planar networkx graph

    kwargs : optional keywords
        See `draw_networkx` for a description of optional keywords.

    Raises
    ------
    NetworkXException
        When `G` is not planar

    Notes
    -----
    The layout is computed each time this function is called.
    For repeated drawing it is much more efficient to call
    `~networkx.drawing.layout.planar_layout` directly and reuse the result::

        >>> G = nx.path_graph(5)
        >>> pos = nx.planar_layout(G)
        >>> nx.draw(G, pos=pos)  # Draw the original graph
        >>> # Draw a subgraph, reusing the same node positions
        >>> nx.draw(G.subgraph([0, 1, 2]), pos=pos, node_color="red")

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.draw_planar(G)

    See Also
    --------
    :func:`~networkx.drawing.layout.planar_layout`
    """
    pass

def apply_alpha(colors, alpha, elem_list, cmap=None, vmin=None, vmax=None):
    """Apply an alpha (or list of alphas) to the colors provided.

    Parameters
    ----------

    colors : color string or array of floats (default='r')
        Color of element. Can be a single color format string,
        or a sequence of colors with the same length as nodelist.
        If numeric values are specified they will be mapped to
        colors using the cmap and vmin,vmax parameters.  See
        matplotlib.scatter for more details.

    alpha : float or array of floats
        Alpha values for elements. This can be a single alpha value, in
        which case it will be applied to all the elements of color. Otherwise,
        if it is an array, the elements of alpha will be applied to the colors
        in order (cycling through alpha multiple times if necessary).

    elem_list : array of networkx objects
        The list of elements which are being colored. These could be nodes,
        edges or labels.

    cmap : matplotlib colormap
        Color map for use if colors is a list of floats corresponding to points
        on a color mapping.

    vmin, vmax : float
        Minimum and maximum values for normalizing colors if a colormap is used

    Returns
    -------

    rgba_colors : numpy ndarray
        Array containing RGBA format values for each of the node colours.

    """
    pass