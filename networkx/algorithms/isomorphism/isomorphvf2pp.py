import networkx as nx
from networkx.utils import groups


class vf2pp:
    def __init__(self, G1, G2, label):
        """Creates the vf2pp object.

        Will let you check if:
            1) G1 is isomorphic to G2
            2) G1 is an induced subgraph of G2
            3) G1 is a subgraph of G2

        Parameters
        ----------
            G1 - Network X Graph Instance
            G2 - Network X Graph Instance
            label - string (the label for which we want to match)
        """
        # Set the self variables to be accessed in other functions
        self.G1 = G1
        self.G2 = G2
        self.V1 = set(G1.nodes())
        self.V2 = set(G2.nodes())
        self.label = label

        # Dictionaries containing the labels of each node
        self.G1_labels = nx.get_node_attributes(G1, label)
        self.G2_labels = nx.get_node_attributes(G2, label)

        # Get the matching order
        self.G1_node_order = self._matching_order()

        # Initilize the mapping -> all values to None
        self.mapping_G1 = {node: None for node in self.V1}
        self.mapping_G2 = {node: None for node in self.V2}

        # Initialize the vectors for storing information
        self.T1 = {node: 0 for node in self.V1}
        self.T1_tilda = {node: 1 for node in self.V1}

        self.T2 = {node: 0 for node in self.V2}
        self.T2_tilda = {node: 1 for node in self.V2}

        self.V1_in_mapping = set()
        self.V2_in_mapping = set()

    ###########################################################################
    #                          Node Order Functions                           #
    ###########################################################################
    def minlist(self, lists_dict):
        """Returns the shortest value of all entries in the dict `lists_dict`

        This is a helper function for later functions.
        `lists_dict` is intended to be a dict of lists of neighbors
        keyed by node. This function finds and returns the shortest
        of these lists.
        """
        return min(lists_dict.values(), key=len)
        # min_list, min_len = None, 0
        # for value in lists_dict.values():
        #    if min_list == None:
        #        min_list, min_len = value, len(value)
        #    else:
        #        if len(value) < min_len:
        #            min_list, min_len = value, len(value)
        # return min_list

    def _process_level(
        self, order, num_in_order, current_depth_nodes, f_m_labels, conn
    ):
        """Helper function for _matching_order,
        returns the updated order, num_in_order, f_m_levels, and conn"""

        while current_depth_nodes:

            # Get the nodes with the max conn
            nodes_by_conn = {}
            for node in current_depth_nodes:
                if conn[node] not in nodes_by_conn:
                    nodes_by_conn[conn[node]] = [node]
                else:
                    nodes_by_conn[conn[node]] += [node]

            max_conn_nodes = nodes_by_conn[max(nodes_by_conn.keys())]

            # Get the max degree from those nodes
            max_deg = None
            max_deg_nodes = None  # this will be "r" in the original paper

            for node in max_conn_nodes:
                if max_deg == None:
                    max_deg = self.G1.degree(node)
                    max_deg_nodes = [node]
                elif self.G1.degree(node) > max_deg:
                    max_deg = self.G1.degree(node)
                    max_deg_nodes = [node]
                elif self.G1.degree(node) == max_deg:
                    max_deg_nodes += [node]

            # Get the right label
            nodes_by_labels = {}

            for node in max_deg_nodes:
                if self.G1_labels[node] not in nodes_by_labels:
                    nodes_by_labels[self.G1_labels[node]] = [node]
                else:
                    nodes_by_labels[self.G1_labels[node]] += [node]

            min_label = None
            min_f_m = None
            for label in nodes_by_labels.keys():
                if min_label == None:
                    min_label = label
                    min_f_m = f_m_labels[label]
                if f_m_labels[label] <= min_f_m:
                    min_label = label
                    min_f_m = f_m_labels[label]

            min_nodes = nodes_by_labels[min_label]

            f_m_labels[min_label] -= 1

            m = min_nodes[0]

            order[num_in_order] = m
            num_in_order += 1

            for node in self.G1.neighbors(m):
                conn[node] += 1

            current_depth_nodes.remove(m)

        return order, num_in_order, f_m_labels, conn

    def _matching_order(self):
        """Returns an order to visit its nodes of G1 in, based on Algorithms 2
        and 3 from the VF2++ paper.
        """
        # Get the set of all nodes so we can remove once they're added:
        V1_not_in_order = self.V1.copy()

        # Create an empty mapping to add to, and an index variable to keep track of where we are
        order = [None for i in range(self.G1.order())]
        num_in_order = 0

        # Number of Neighbors of each node in the current node order (Used to determine node order)
        conn = {node: 0 for node in self.V1}

        while len(V1_not_in_order) > 0:

            # Get the labels of every node not in the order yet
            current_labels = {}
            for node in V1_not_in_order:
                current_labels[node] = self.G1_labels[node]

            # Get the nodes with the rarest label
            min_nodes = self.minlist(groups(current_labels))

            # Get the node from this list with the highest degree
            max_degree = None
            max_node = None  # this will be "r" in the original paper

            for node in min_nodes:
                if max_degree == None:
                    max_degree = self.G1.degree(node)
                    max_node = node
                elif self.G1.degree(node) > max_degree:
                    max_degree = self.G1.degree(node)
                    max_node = node

            # Add the root node of this component
            order[num_in_order] = max_node
            V1_not_in_order.discard(max_node)
            num_in_order += 1

            # Initializing F_M(l) to be updated later, need the rarety of each label in V2
            f_m_labels = {}

            node_by_label_G2 = groups(self.G2_labels)
            for a_label in node_by_label_G2:
                f_m_labels[a_label] = len(node_by_label_G2[a_label])

            # look at each depth
            T_edges = list(nx.bfs_edges(self.G1, max_node))
            current_depth_nodes = []

            for node_num in range(len(T_edges)):

                if (
                    T_edges[node_num][0] not in current_depth_nodes
                ):  # Keep going until you get the entire depth
                    current_depth_nodes += [T_edges[node_num][1]]

                else:  # process this level and then initalize the next one

                    # first remove all the nodes we will use:
                    V1_not_in_order = set(V1_not_in_order) - set(current_depth_nodes)
                    order, num_in_order, f_m_labels, conn = self._process_level(
                        order, num_in_order, current_depth_nodes, f_m_labels, conn
                    )

                    current_depth_nodes = [T_edges[node_num][1]]

            # Process the last level
            V1_not_in_order = set(V1_not_in_order) - set(current_depth_nodes)

            order, num_in_order, f_m_labels, conn = self._process_level(
                order, num_in_order, current_depth_nodes, f_m_labels, conn
            )

        return order

    ###########################################################################
    #                        Consistency Functions                            #
    ###########################################################################

    def _cons_is_i(self, u, v):
        """
        Determines if a point is consistent with a mapping for an induced subgraph
        or isomorphism problem. Returns a boolean, True if it is consistent and
        False if it is not.

        Assumes the current mapping is consistent, in this implementation we will
        only be creating consistent mappings.

        Parameters:
            u: node from G1
            v: node from G2
                This function checks if mapping u (from G1) to v (from G2) would be
                consistent with the current mapping m
        """

        # Check if the two nodes have the same value for label
        if self.G1_labels[u] != self.G2_labels[v]:
            return False

        # Check that all the neighbors of v in the range of the current mapping
        # have an edge between u and the node they are mapped to

        for neighbor in set(self.G2.neighbors(v)):
            if self.mapping_G2[neighbor] != None:
                if not self.G1.has_edge(u, self.mapping_G2[neighbor]):
                    return False

        # Check that all the neighbors of u in the domain of the current mapping
        # have an edge between v and the node they are mapped to

        for neighbor in set(self.G1.neighbors(u)):
            if self.mapping_G1[neighbor] != None:
                if not self.G2.has_edge(v, self.mapping_G1[neighbor]):
                    return False

        # Return True if all of these tests are passed
        return True

    def _cons_s(self, u, v):
        """
        Determines if a point is consistent with a mapping for an subgraph problem.
        Returns a boolean, True if it is consistent and False if it is not.

        Parameters:
            u: node from G1
            v: node from G2
                This function checks if mapping u (from G1) to v (from G2) would be
                consistent with the current mapping m
        """
        # Check if the two nodes have the same value for label
        if self.G1_labels[u] != self.G2_labels[v]:
            return False

        # Check that all the neighbors of u in the domain of the current mapping
        # have an edge between v and the node they are mapped to

        for neighbor in set(self.G1.neighbors(u)):
            if self.mapping_G1[neighbor] != None:
                if not self.G2.has_edge(u, self.mapping_G1[neighbor]):
                    return False

        # Return True if all of these tests are passed
        return True

    ###########################################################################
    #                           Cutting Functions                             #
    ###########################################################################

    def _cut_i(self, u, v):
        """Determines whether the adding the pair (u,v) could possibly result
        in an isomorphism. Returns a boolean, True if there is no possible
        mapping with the current mapping and (u,v), False otherwise.

        Parameters:
            u: node from G1
            v: node from G2
        """

        u_neighbors = set(self.G1.neighbors(u))
        v_neighbors = set(self.G2.neighbors(v))

        u_n_in_t = {}
        u_n_in_t_tilda = {}

        v_n_in_t = {}
        v_n_in_t_tilda = {}

        for node in u_neighbors:
            if self.T1[node] > 0:
                if self.G1_labels[node] not in u_n_in_t:
                    u_n_in_t[self.G1_labels[node]] = 1
                else:
                    u_n_in_t[self.G1_labels[node]] += 1
            if self.T1_tilda[node] == 1:
                if self.G1_labels[node] not in u_n_in_t_tilda:
                    u_n_in_t_tilda[self.G1_labels[node]] = 1
                else:
                    u_n_in_t_tilda[self.G1_labels[node]] += 1

        for node in v_neighbors:
            if self.T2[node] > 0:
                if self.G2_labels[node] not in v_n_in_t:
                    v_n_in_t[self.G2_labels[node]] = 1
                else:
                    v_n_in_t[self.G2_labels[node]] += 1
            if self.T2_tilda[node] == 1:
                if self.G2_labels[node] not in v_n_in_t_tilda:
                    v_n_in_t_tilda[self.G2_labels[node]] = 1
                else:
                    v_n_in_t_tilda[self.G2_labels[node]] += 1

        for label in u_n_in_t:
            if label not in v_n_in_t:
                return True
            elif u_n_in_t[label] != v_n_in_t[label]:
                return True

        for label in u_n_in_t_tilda:
            if label not in v_n_in_t_tilda:
                return True
            elif u_n_in_t_tilda[label] != v_n_in_t_tilda[label]:
                return True

        return False

    def _cut_is(self, u, v):
        """Determines whether the adding the pair (u,v) could possibly result
        in an induced subgraph. Returns a boolean, True if there is no possible
        mapping with the current mapping and (u,v), False otherwise.

        Parameters:
            u: node from G1
            v: node from G2
        """
        u_neighbors = set(self.G1.neighbors(u))
        v_neighbors = set(self.G2.neighbors(v))

        u_n_in_t = {}
        u_n_in_t_tilda = {}

        v_n_in_t = {}
        v_n_in_t_tilda = {}

        for node in u_neighbors:
            if self.T1[node] > 0:
                if self.G1_labels[node] not in u_n_in_t:
                    u_n_in_t[self.G1_labels[node]] = 1
                else:
                    u_n_in_t[self.G1_labels[node]] += 1
            if self.T1_tilda[node] == 1:
                if self.G1_labels[node] not in u_n_in_t_tilda:
                    u_n_in_t_tilda[self.G1_labels[node]] = 1
                else:
                    u_n_in_t_tilda[self.G1_labels[node]] += 1

        for node in v_neighbors:
            if self.T2[node] > 0:
                if self.G2_labels[node] not in v_n_in_t:
                    v_n_in_t[self.G2_labels[node]] = 1
                else:
                    v_n_in_t[self.G2_labels[node]] += 1
            if self.T2_tilda[node] == 1:
                if self.G2_labels[node] not in v_n_in_t_tilda:
                    v_n_in_t_tilda[self.G2_labels[node]] = 1
                else:
                    v_n_in_t_tilda[self.G2_labels[node]] += 1

        for label in u_n_in_t:
            if label not in v_n_in_t:
                return True
            elif u_n_in_t[label] > v_n_in_t[label]:
                return True

        for label in u_n_in_t_tilda:
            if label not in v_n_in_t_tilda:
                return True
            elif u_n_in_t_tilda[label] > v_n_in_t_tilda[label]:
                return True

        return False

    def _cut_s(self, u, v):
        """Determines whether the adding the pair (u,v) could possibly result
        in an induced subgraph. Returns a boolean, True if there is no possible
        mapping with the current mapping and (u,v), False otherwise.

        Parameters:
            u: node from G1
            v: node from G2
        """
        u_neighbors = set(self.G1.neighbors(u))
        v_neighbors = set(self.G2.neighbors(v))

        u_n_in_t = {}
        v_n_in_t = {}

        for node in u_neighbors:
            if self.T1[node] > 0:
                if self.G1_labels[node] not in u_n_in_t:
                    u_n_in_t[self.G1_labels[node]] = 1
                else:
                    u_n_in_t[self.G1_labels[node]] += 1

        for node in v_neighbors:
            if self.T2[node] > 0:
                if self.G2_labels[node] not in v_n_in_t:
                    v_n_in_t[self.G2_labels[node]] = 1
                else:
                    v_n_in_t[self.G2_labels[node]] += 1

        for label in u_n_in_t:
            if label not in v_n_in_t:
                return True
            elif u_n_in_t[label] > v_n_in_t[label]:
                return True

        return False

    ###########################################################################
    #                            Candidate Pairs                              #
    ###########################################################################

    def _cand_pairs(self, u):
        """Gives a set of nodes in G2, which u could possibly be mapped to.

        Parameters:
            u - Node from G1

        """
        u_label = self.G1_labels[u]

        # If G2 has no nodes with u's label, return an empty list
        if u_label not in groups(self.G2_labels):
            return []

        # Get everything with the right label then remove those that've already been mapped
        possible_v = set(groups(self.G2_labels)[u_label])  # Things with the right label
        to_remove = []
        for node in possible_v:
            if self.mapping_G2[node] != None:
                to_remove += [node]

        possible_v = possible_v - set(to_remove)

        # Get the covered neighbors of u in G1, and the nodes they map to in G2
        covered_neighbors_u = []
        covered_neighbors_v = []

        for node in self.G1.neighbors(u):
            if self.mapping_G1[node] != None:
                covered_neighbors_u += [node]
                covered_neighbors_v += [self.mapping_G1[node]]

        # Get all v in V2 s.t. if a neighbor ~u~ of u is covered, it is mapped to a neighbor ~v~ of v
        poss_2 = []
        for node in self.G1.neighbors(u):
            if self.mapping_G1[node] != None:
                v = self.mapping_G1[node]
                if (self.mapping_G2[v] == None) & set(covered_neighbors_v).issubset(
                    set(self.G2.neighbors(v))
                ):
                    poss_2 += [v]

        # If there are no neighbors that work, return the things with the right label (this could happen if it has no covered neighbors)
        if poss_2 == []:
            return possible_v

        # Otherwise take the intersection to get the neighbors that have the right label
        final_candidates = possible_v.intersection(set(poss_2))

        return list(final_candidates)

    ###########################################################################
    #                   Functions to Call to Check Things                     #
    ###########################################################################

    def _is_isomorphic(self):
        """Starts of the isomorphic check. Uses _is_isomorphic_helper"""
        return self._is_isomorphic_helper(0)

    def _is_isomorphic_helper(self, depth):
        """Helper function for _is_isomorphic,

        Parameter
        ---------
        depth - int:
            the index of the node in G1_node_order we are currently trying to match
        """

        # Is the mapping complete?
        if depth == (len(self.G1_node_order)):
            return True

        # Get the node we are currently trying to match
        u = self.G1_node_order[depth]

        # Get the candidate pairs for this node
        poss_v = self._cand_pairs(u)

        # Remove candidates with wrong degree
        to_remove = []
        for v in poss_v:
            if nx.degree(self.G1, u) != nx.degree(self.G2, v):
                to_remove.append(v)

        poss_v = list(set(poss_v) - set(to_remove))

        # Make sure there is a candidate
        if poss_v == []:
            return False

        # Go through the remaining candidates one by one
        for v in poss_v:

            # Check the criteria
            if self._cons_is_i(u, v) & (not self._cut_i(u, v)):

                # Temporarily extend the matching
                self.mapping_G1[u] = v
                self.mapping_G2[v] = u

                # Temporarily update the sets
                for node in self.G1.neighbors(u):
                    self.T1[node] += 1
                    self.T1_tilda[node] -= 1

                for node in self.G2.neighbors(v):
                    self.T2[node] += 1
                    self.T2_tilda[node] -= 1

                self.T1[u] = self.T1[u] * (-1)
                self.T1_tilda[u] -= 1
                self.T2[v] = self.T2[v] * (-1)
                self.T2_tilda[v] -= 1

                # Check if it continues from there (recursive call)
                if self._is_isomorphic_helper(depth + 1):
                    return True

                # If not unmatch it and revert the sets
                else:
                    self.mapping_G1[u] = None
                    self.mapping_G2[v] = None

                    for node in self.G1.neighbors(u):
                        self.T1[node] -= 1
                        self.T1_tilda[node] += 1

                    for node in self.G2.neighbors(v):
                        self.T2[node] -= 1
                        self.T2_tilda[node] += 1

                    self.T1[u] = self.T1[u] * (-1)
                    self.T1_tilda[u] += 1
                    self.T2[v] = self.T2[v] * (-1)
                    self.T2_tilda[v] += 1

        # If none of the candidate pairs worked then return False
        return False

    def _is_induced_subgraph(self):
        """Starts of the induced subgraph check. Uses _is_is_helper"""
        return self._is_is_helper(0)

    def _is_is_helper(self, depth):
        """Helper function for _is_induced_subgraph,

        Parameter
        ---------
        depth - int:
            the index of the node in G1_node_order we are currently trying to match
        """

        # Is the mapping complete?
        if depth == (len(self.G1_node_order)):
            return True

        # Get the node we are currently trying to match
        u = self.G1_node_order[depth]

        # Get the candidate pairs for this node
        poss_v = self._cand_pairs(u)

        # Remove candidates with wrong degree
        to_remove = []
        for v in poss_v:
            if nx.degree(self.G1, u) > nx.degree(self.G2, v):
                to_remove.append(v)

        poss_v = list(set(poss_v) - set(to_remove))

        # Make sure there is a candidate
        if poss_v == []:
            return False

        # Go through the remaining candidates one by one
        for v in poss_v:

            # Check the criteria
            if self._cons_is_i(u, v) & (not self._cut_is(u, v)):

                # Temporarily extend the matching
                self.mapping_G1[u] = v
                self.mapping_G2[v] = u

                # Temporarily update the sets
                for node in self.G1.neighbors(u):
                    self.T1[node] += 1
                    self.T1_tilda[node] -= 1

                for node in self.G2.neighbors(v):
                    self.T2[node] += 1
                    self.T2_tilda[node] -= 1

                self.T1[u] = self.T1[u] * (-1)
                self.T1_tilda[u] -= 1
                self.T2[v] = self.T2[v] * (-1)
                self.T2_tilda[v] -= 1

                # Check if it continues from there (recursive call)
                if self._is_is_helper(depth + 1):
                    return True

                # If not unmatch it and revert the sets
                else:
                    self.mapping_G1[u] = None
                    self.mapping_G2[v] = None

                    for node in self.G1.neighbors(u):
                        self.T1[node] -= 1
                        self.T1_tilda[node] += 1

                    for node in self.G2.neighbors(v):
                        self.T2[node] -= 1
                        self.T2_tilda[node] += 1

                    self.T1[u] = self.T1[u] * (-1)
                    self.T1_tilda[u] += 1
                    self.T2[v] = self.T2[v] * (-1)
                    self.T2_tilda[v] += 1

        # If none of the candidate pairs worked then return False
        return False

    def _is_subgraph(self):
        """Starts of the subgraph check. Uses _is_s_helper"""
        return self._is_s_helper(0)

    def _is_s_helper(self, depth):
        """Helper function for _is_subgraph,

        Parameter
        ---------
        depth - int:
            the index of the node in G1_node_order we are currently trying to match
        """
        # Is the mapping complete?
        if depth == (len(self.G1_node_order)):
            return True

        # Get the node we are currently trying to match
        u = self.G1_node_order[depth]

        # Get the candidate pairs for this node
        poss_v = self._cand_pairs(u)

        # Remove candidates with wrong degree

        to_remove = []
        for v in poss_v:
            if nx.degree(self.G1, u) > nx.degree(self.G2, v):
                to_remove.append(v)

        poss_v = list(set(poss_v) - set(to_remove))

        # Make sure there is a candidate
        if poss_v == []:
            return False

        # Go through the remaining candidates one by one
        for v in poss_v:
            # Check the criteria
            if self._cons_s(u, v) & (not self._cut_s(u, v)):

                # Temporarily extend the matching
                self.mapping_G1[u] = v
                self.mapping_G2[v] = u

                # Temporarily update the sets
                for node in self.G1.neighbors(u):
                    self.T1[node] += 1
                    self.T1_tilda[node] -= 1

                for node in self.G2.neighbors(v):
                    self.T2[node] += 1
                    self.T2_tilda[node] -= 1

                self.T1[u] = self.T1[u] * (-1)
                self.T1_tilda[u] -= 1
                self.T2[v] = self.T2[v] * (-1)
                self.T2_tilda[v] -= 1

                # Check if it continues from there (recursive call)
                if self._is_s_helper(depth + 1):
                    return True

                # If not unmatch it and revert the sets
                else:
                    self.mapping_G1[u] = None
                    self.mapping_G2[v] = None

                    for node in self.G1.neighbors(u):
                        self.T1[node] -= 1
                        self.T1_tilda[node] += 1

                    for node in self.G2.neighbors(v):
                        self.T2[node] -= 1
                        self.T2_tilda[node] += 1

                    self.T1[u] = self.T1[u] * (-1)
                    self.T1_tilda[u] += 1
                    self.T2[v] = self.T2[v] * (-1)
                    self.T2_tilda[v] += 1

        # If none of the candidate pairs worked then return False
        return False
