"""Read and write graphs in GEXF format.

.. warning::
    This parser uses the standard xml library present in Python, which is
    insecure - see :external+python:mod:`xml` for additional information.
    Only parse GEFX files you trust.

GEXF (Graph Exchange XML Format) is a language for describing complex
network structures, their associated data and dynamics.

This implementation does not support mixed graphs (directed and
undirected edges together).

Format
------
GEXF is an XML format.  See http://gexf.net/schema.html for the
specification and http://gexf.net/basic.html for examples.
"""
import itertools
import time
from xml.etree.ElementTree import Element, ElementTree, SubElement, register_namespace, tostring
import networkx as nx
from networkx.utils import open_file
__all__ = ['write_gexf', 'read_gexf', 'relabel_gexf_graph', 'generate_gexf']

@open_file(1, mode='wb')
def write_gexf(G, path, encoding='utf-8', prettyprint=True, version='1.2draft'):
    """Write G in GEXF format to path.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Node attributes are checked according to the version of the GEXF
    schemas used for parameters which are not user defined,
    e.g. visualization 'viz' [2]_. See example for usage.

    Parameters
    ----------
    G : graph
       A NetworkX graph
    path : file or string
       File or file name to write.
       File names ending in .gz or .bz2 will be compressed.
    encoding : string (optional, default: 'utf-8')
       Encoding for text data.
    prettyprint : bool (optional, default: True)
       If True use line breaks and indenting in output XML.
    version: string (optional, default: '1.2draft')
       The version of GEXF to be used for nodes attributes checking

    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> nx.write_gexf(G, "test.gexf")

    # visualization data
    >>> G.nodes[0]["viz"] = {"size": 54}
    >>> G.nodes[0]["viz"]["position"] = {"x": 0, "y": 1}
    >>> G.nodes[0]["viz"]["color"] = {"r": 0, "g": 0, "b": 256}


    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    The node id attribute is set to be the string of the node label.
    If you want to specify an id use set it as node data, e.g.
    node['a']['id']=1 to set the id of node 'a' to 1.

    References
    ----------
    .. [1] GEXF File Format, http://gexf.net/
    .. [2] GEXF schema, http://gexf.net/schema.html
    """
    pass

def generate_gexf(G, encoding='utf-8', prettyprint=True, version='1.2draft'):
    """Generate lines of GEXF format representation of G.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Parameters
    ----------
    G : graph
    A NetworkX graph
    encoding : string (optional, default: 'utf-8')
    Encoding for text data.
    prettyprint : bool (optional, default: True)
    If True use line breaks and indenting in output XML.
    version : string (default: 1.2draft)
    Version of GEFX File Format (see http://gexf.net/schema.html)
    Supported values: "1.1draft", "1.2draft"


    Examples
    --------
    >>> G = nx.path_graph(4)
    >>> linefeed = chr(10)  # linefeed=

    >>> s = linefeed.join(nx.generate_gexf(G))
    >>> for line in nx.generate_gexf(G):  # doctest: +SKIP
    ...     print(line)

    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    The node id attribute is set to be the string of the node label.
    If you want to specify an id use set it as node data, e.g.
    node['a']['id']=1 to set the id of node 'a' to 1.

    References
    ----------
    .. [1] GEXF File Format, https://gephi.org/gexf/format/
    """
    pass

@open_file(0, mode='rb')
@nx._dispatchable(graphs=None, returns_graph=True)
def read_gexf(path, node_type=None, relabel=False, version='1.2draft'):
    """Read graph in GEXF format from path.

    "GEXF (Graph Exchange XML Format) is a language for describing
    complex networks structures, their associated data and dynamics" [1]_.

    Parameters
    ----------
    path : file or string
       File or file name to read.
       File names ending in .gz or .bz2 will be decompressed.
    node_type: Python type (default: None)
       Convert node ids to this type if not None.
    relabel : bool (default: False)
       If True relabel the nodes to use the GEXF node "label" attribute
       instead of the node "id" attribute as the NetworkX node label.
    version : string (default: 1.2draft)
    Version of GEFX File Format (see http://gexf.net/schema.html)
       Supported values: "1.1draft", "1.2draft"

    Returns
    -------
    graph: NetworkX graph
        If no parallel edges are found a Graph or DiGraph is returned.
        Otherwise a MultiGraph or MultiDiGraph is returned.

    Notes
    -----
    This implementation does not support mixed graphs (directed and undirected
    edges together).

    References
    ----------
    .. [1] GEXF File Format, http://gexf.net/
    """
    pass

class GEXF:
    versions = {'1.1draft': {'NS_GEXF': 'http://www.gexf.net/1.1draft', 'NS_VIZ': 'http://www.gexf.net/1.1draft/viz', 'NS_XSI': 'http://www.w3.org/2001/XMLSchema-instance', 'SCHEMALOCATION': ' '.join(['http://www.gexf.net/1.1draft', 'http://www.gexf.net/1.1draft/gexf.xsd']), 'VERSION': '1.1'}, '1.2draft': {'NS_GEXF': 'http://www.gexf.net/1.2draft', 'NS_VIZ': 'http://www.gexf.net/1.2draft/viz', 'NS_XSI': 'http://www.w3.org/2001/XMLSchema-instance', 'SCHEMALOCATION': ' '.join(['http://www.gexf.net/1.2draft', 'http://www.gexf.net/1.2draft/gexf.xsd']), 'VERSION': '1.2'}}
    convert_bool = {'true': True, 'false': False, 'True': True, 'False': False, '0': False, 0: False, '1': True, 1: True}

class GEXFWriter(GEXF):

    def __init__(self, graph=None, encoding='utf-8', prettyprint=True, version='1.2draft'):
        self.construct_types()
        self.prettyprint = prettyprint
        self.encoding = encoding
        self.set_version(version)
        self.xml = Element('gexf', {'xmlns': self.NS_GEXF, 'xmlns:xsi': self.NS_XSI, 'xsi:schemaLocation': self.SCHEMALOCATION, 'version': self.VERSION})
        meta_element = Element('meta')
        subelement_text = f'NetworkX {nx.__version__}'
        SubElement(meta_element, 'creator').text = subelement_text
        meta_element.set('lastmodifieddate', time.strftime('%Y-%m-%d'))
        self.xml.append(meta_element)
        register_namespace('viz', self.NS_VIZ)
        self.edge_id = itertools.count()
        self.attr_id = itertools.count()
        self.all_edge_ids = set()
        self.attr = {}
        self.attr['node'] = {}
        self.attr['edge'] = {}
        self.attr['node']['dynamic'] = {}
        self.attr['node']['static'] = {}
        self.attr['edge']['dynamic'] = {}
        self.attr['edge']['static'] = {}
        if graph is not None:
            self.add_graph(graph)

    def __str__(self):
        if self.prettyprint:
            self.indent(self.xml)
        s = tostring(self.xml).decode(self.encoding)
        return s

class GEXFReader(GEXF):

    def __init__(self, node_type=None, version='1.2draft'):
        self.construct_types()
        self.node_type = node_type
        self.simple_graph = True
        self.set_version(version)

    def __call__(self, stream):
        self.xml = ElementTree(file=stream)
        g = self.xml.find(f'{{{self.NS_GEXF}}}graph')
        if g is not None:
            return self.make_graph(g)
        for version in self.versions:
            self.set_version(version)
            g = self.xml.find(f'{{{self.NS_GEXF}}}graph')
            if g is not None:
                return self.make_graph(g)
        raise nx.NetworkXError('No <graph> element in GEXF file.')

def relabel_gexf_graph(G):
    """Relabel graph using "label" node keyword for node label.

    Parameters
    ----------
    G : graph
       A NetworkX graph read from GEXF data

    Returns
    -------
    H : graph
      A NetworkX graph with relabeled nodes

    Raises
    ------
    NetworkXError
        If node labels are missing or not unique while relabel=True.

    Notes
    -----
    This function relabels the nodes in a NetworkX graph with the
    "label" attribute.  It also handles relabeling the specific GEXF
    node attributes "parents", and "pid".
    """
    pass