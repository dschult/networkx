from collections import Counter, defaultdict
import operator
import networkx as nx
from networkx.utils import groups


__all__ = [
    "is_isomorphic",
    "is_isomorphic_recursive",
    "is_induced_subgraph_isomorphic",
    "is_induced_subgraph_isomorphic_recursive",
    "is_subgraph_monomorphic",
    "is_subgraph_monomorphic_recursive",
]


class GraphMatcher_vf2pp:
    def __init__(self, G1, G2, label):
        """Creates the GraphMatcher_vf2pp object.

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
        self.label = label
        self.G1_degree = dict(G1.degree)
        self.G2_degree = dict(G2.degree)

        # Dictionaries containing the labels of each node
        self.G1_labels = nx.get_node_attributes(G1, label)
        self.G2_labels = nx.get_node_attributes(G2, label)
        self.nodes_by_G1labels = groups(self.G1_labels)
        self.nodes_by_G2labels = groups(self.G2_labels)

        # Get the matching order
        self.G1_node_order = self._matching_order()
        # print("node order", self.G1_node_order)

        # Initilize the mapping -> all values to None
        self.mapping_G1 = {node: None for node in G1}
        self.mapping_G2 = {node: None for node in G2}

        # Initialize the vectors for storing information
        self.T1 = {node: 0 for node in G1}
        self.T1_tilda = {node: 1 for node in G1}

        self.T2 = {node: 0 for node in G2}
        self.T2_tilda = {node: 1 for node in G2}

        # print(f"G1: {G1} labels: {self.G1_labels}")
        # print(f"G2: {G2} labels: {self.G2_labels}")
        # print(f"G2.nodes: {G2.nodes.data('label')}")

    #######################################################################
    #                      Node Order Methods                             #
    #######################################################################
    def _process_level(self, order, current_nodes, label_rarity, used_degree):
        """Update order, label_rarity and used_degree

        Helper function for _matching_order

        returns the updated order, label_rarity, and used_degree
        """
        while current_nodes:
            # Get the nodes with the max used_degree
            max_used_deg = -1
            for node in current_nodes:
                deg = used_degree[node]
                if deg >= max_used_deg:  # most common case: deg < max_deg
                    if deg > max_used_deg:
                        max_used_deg = deg
                        max_nodes = [node]
                    else:  # deg == max_deg
                        max_nodes.append(node)

            # Get the max_used_degree node with the rarest label
            next_node = min(max_nodes, key=lambda x: label_rarity[self.G1_labels[x]])
            order.append(next_node)

            for node in self.G1.neighbors(next_node):
                used_degree[node] += 1

            current_nodes.remove(next_node)
            label_rarity[self.G1_labels[next_node]] -= 1

    def _matching_order(self):
        """Returns an order in which to visit the nodes of G1

        Based on Algorithms 2 and 3 from the VF2++ paper.
        """
        # Get the set of all nodes so we can remove once they're added:
        V1_not_in_order = set(self.G1)

        # Create an empty mapping to add to, and an index variable to keep track of where we are
        order = []

        # Number of Neighbors of each node in the current node order (Used to determine node order)
        used_degree = {node: 0 for node in self.G1}

        while V1_not_in_order:
            # Get the labels of every node not in the order yet
            current_labels = {node: self.G1_labels[node] for node in V1_not_in_order}

            # Get the nodes with the rarest label
            # groups() returns a dict keyed by label to the set of nodes with that label
            rare_nodes = min(groups(current_labels).values(), key=len)

            # Get the node from this list with the highest degree
            max_node = max(rare_nodes, key=self.G1.degree)

            # Add the root node of this component
            order.append(max_node)
            V1_not_in_order.discard(max_node)

            # Initializing F_M(l) = label_rarity to be updated later, need the rarety of each label in V2
            label_rarity = {
                a_label: len(nodes) for a_label, nodes in self.nodes_by_G2labels.items()
            }

            # consider nodes at each depth from max_node
            current_nodes = set()
            for node, nbr in nx.bfs_edges(self.G1, max_node):
                if node not in current_nodes:
                    current_nodes.add(nbr)
                else:
                    # first remove all the nodes we will use:
                    V1_not_in_order -= current_nodes
                    # process current level's nodes
                    self._process_level(order, current_nodes, label_rarity, used_degree)
                    # initialize next level
                    current_nodes = {nbr}

            # Process the last level
            V1_not_in_order -= current_nodes
            self._process_level(order, current_nodes, label_rarity, used_degree)

        return order

    #######################################################################
    #                    Consistency Functions                            #
    #######################################################################
    def _can_be_isomorphism(self, u, v):
        """Is mapping u -> v consistent for an induced subgraph or isomorphism

        This function checks if mapping u (from G1) to v (from G2) would be
        consistent with the current mapping m for an induced subgraph or
        isomorphism problem.

        Returns True if consistent and False otherwise.

        Assumes the current mapping is consistent, in this implementation we will
        only be creating consistent mappings.

        Parameters:
            u: node from G1
            v: node from G2
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

    def _can_be_monomorphism(self, u, v):
        """Is mapping u -> v consistent for a monomorphism (subgraph) problem

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

    #######################################################################
    #                       Cutting Functions                             #
    #######################################################################
    def _cant_be_isomorphism(self, u, v):
        """Find if adding u -> v makes the mapping not possibly an isomorphism.

        If adding u -> v to the mapping could possibly result in an isomorphism
        return False.  Returns True if this part of the map cannot be included.

        Returns True if there is no possible mapping after adding, False otherwise.

        Parameters:
            u: node from G1
            v: node from G2
        """
        u_n_in_t = Counter()
        u_n_in_t_tilda = Counter()

        v_n_in_t = Counter()
        v_n_in_t_tilda = Counter()

        for node in self.G1.neighbors(u):
            if self.T1[node] > 0:
                u_n_in_t[self.G1_labels[node]] += 1
            if self.T1_tilda[node] == 1:
                u_n_in_t_tilda[self.G1_labels[node]] += 1

        for node in self.G2.neighbors(v):
            if self.T2[node] > 0:
                v_n_in_t[self.G2_labels[node]] += 1
            if self.T2_tilda[node] == 1:
                v_n_in_t_tilda[self.G2_labels[node]] += 1

        for label, numb_nbrs in u_n_in_t.items():
            if label not in v_n_in_t or v_n_in_t[label] != numb_nbrs:
                return True

        for label, numb_nbrs in u_n_in_t_tilda.items():
            if label not in v_n_in_t_tilda or v_n_in_t_tilda[label] != numb_nbrs:
                return True

        return False

    def _cant_be_ind_subg_iso(self, u, v):
        """Determines whether the adding the pair (u,v) could possibly result
        in an induced subgraph. Returns a boolean, True if there is no possible
        mapping with the current mapping and (u,v), False otherwise.

        Parameters:
            u: node from G1
            v: node from G2
        """
        u_n_in_t = Counter()
        u_n_in_t_tilda = Counter()

        v_n_in_t = Counter()
        v_n_in_t_tilda = Counter()

        for node in self.G1.neighbors(u):
            if self.T1[node] > 0:
                u_n_in_t[self.G1_labels[node]] += 1
            if self.T1_tilda[node] == 1:
                u_n_in_t_tilda[self.G1_labels[node]] += 1

        for node in self.G2.neighbors(v):
            if self.T2[node] > 0:
                v_n_in_t[self.G2_labels[node]] += 1
            if self.T2_tilda[node] == 1:
                # print(f"G2 neighbors of {v}: {list(self.G2.neighbors(v))}")
                # print(f"What is going on here? node:{node} labels:{self.G2_labels}")
                v_n_in_t_tilda[self.G2_labels[node]] += 1

        for label, numb_nbrs in u_n_in_t.items():
            if label not in v_n_in_t or numb_nbrs > v_n_in_t[label]:
                return True

        for label, numb_nbrs in u_n_in_t_tilda.items():
            if label not in v_n_in_t_tilda or numb_nbrs > v_n_in_t_tilda[label]:
                return True

        return False

    def _cant_be_monomorphism(self, u, v):
        """Determines whether the adding the pair (u,v) could possibly result
        in a subgraph. Returns a boolean, True if there is no possible
        mapping with the current mapping and (u,v), False otherwise.

        Parameters:
            u: node from G1
            v: node from G2
        """
        u_n_in_t = Counter()
        v_n_in_t = Counter()

        for node in self.G1.neighbors(u):
            if self.T1[node] > 0:
                u_n_in_t[self.G1_labels[node]] += 1

        for node in self.G2.neighbors(v):
            if self.T2[node] > 0:
                v_n_in_t[self.G2_labels[node]] += 1

        for label, numb_nbrs in u_n_in_t.items():
            if label not in v_n_in_t or numb_nbrs > v_n_in_t[label]:
                return True

        return False

    #######################################################################
    #                        Candidate Pairs                              #
    #######################################################################
    def _cand_pairs(self, u):
        """Gives a set of nodes in G2, which u could possibly be mapped to.

        Parameters:
            u - Node from G1

        """
        u_label = self.G1_labels[u]
        u_nbrs = self.G1.neighbors(u)

        G2_groups = groups(self.G2_labels)

        # If G2 has no nodes with u's label, return an empty list
        if u_label not in G2_groups:
            return []

        # Get nodes with the right label that have not been mapped
        possible_v = {n for n in G2_groups[u_label] if self.mapping_G2[n] is None}

        # Get the covered neighbors of u in G1, and the nodes they map to in G2
        covered_nbrs_v = {
            self.mapping_G1[node]
            for node in u_nbrs
            if self.mapping_G1[node] is not None
        }

        # Get all v in G2 s.t. if u_nbr of u is covered, it is mapped to a v_nbr of v
        possible_v2 = {
            v
            for v in covered_nbrs_v
            if self.mapping_G2[v] is None
            if covered_nbrs_v.issubset(self.G2.neighbors(v))
        }

        if possible_v2:
            return possible_v2.intersection(possible_v)

        # If there are no neighbors that work (e.g. no covered nodes exist)
        # return the nodes with the right label.
        return possible_v

    #####################################################################
    #               Functions to Call to Check Things                   #
    #####################################################################
    def is_isomorphic(self):
        return self.is_isomorphic_any(
            self._can_be_isomorphism, self._cant_be_isomorphism, operator.eq
        )

    def is_subgraph_monomorphic(self):
        return self.is_isomorphic_any(
            self._can_be_monomorphism, self._cant_be_monomorphism, operator.ge
        )

    def is_induced_subgraph_isomorphic(self):
        return self.is_isomorphic_any(
            self._can_be_isomorphism, self._cant_be_ind_subg_iso, operator.ge
        )

    def is_isomorphic_any(self, consistency, cut, degree_check):
        # for slight speed increase
        G1_degree = self.G1_degree
        G2_degree = self.G2_degree
        G1_neighbors = self.G1.neighbors
        G2_neighbors = self.G2.neighbors
        mapping_G1 = self.mapping_G1
        mapping_G2 = self.mapping_G2
        T1, T1_tilda = self.T1, self.T1_tilda
        T2, T2_tilda = self.T2, self.T2_tilda
        G1_node_order = self.G1_node_order
        cand_pairs = self._cand_pairs

        # now set up the main loop
        N = len(self.G1)
        depth = 0
        stack = []

        # Main loop
        while depth < N:
            if depth == len(stack):
                u = G1_node_order[depth]
                u_degree = G1_degree[u]
                poss_v = [
                    v
                    for v in cand_pairs(u)
                    if degree_check(G2_degree[v], u_degree)
                    if consistency(u, v)
                    if not cut(u, v)
                ]
                stack.append((u, poss_v))
            else:
                u, poss_v = stack[-1]
            # explore the next image of u if there is one
            if poss_v:
                v = poss_v[0]  # next(poss_v)
                if v is None:
                    print("v is None!")
                    raise Error("Why is v None?")
                    # tmp=list(poss_v)
                    # print(f"u:{u}, poss_v:{tmp}")
                    # print(stack)
                    # poss_v=iter(tmp)
                    # stack[-1][-1]=poss_v
                # Temporarily extend the matching
                mapping_G1[u] = v
                mapping_G2[v] = u

                # Temporarily update the T sets
                for node in G1_neighbors(u):
                    T1[node] += 1
                    T1_tilda[node] -= 1

                for node in G2_neighbors(v):
                    T2[node] += 1
                    T2_tilda[node] -= 1

                T1[u] = T1[u] * (-1)
                T1_tilda[u] -= 1
                T2[v] = T2[v] * (-1)
                T2_tilda[v] -= 1

                # go to the next node
                depth += 1
                # print(f"Processed ({u}, {v}) and moving to depth {depth}")
                # print(f"mapping: {mapping_G1}")
                # print(f"Stack: {stack}")
                poss_v.pop(0)
                #### printing debugging stuff only
                # astack = []
                # print("Stack: ", end="")
                # for n, nbrs in stack:
                #     tmp = list(nbrs)
                #     astack.append((n, iter(tmp)))
                #     print((n, tmp), end="")
                # print()
                # stack = astack
                ####

            # out of poss_v go back a level.
            else:
                # get v, unmatch it and revert the sets
                v = mapping_G1[u]
                # print(f"new v {v}")
                while v is None:
                    depth -= 1
                    if len(stack) == 1:
                        #print("No isomorphism")
                        return False
                    # print("prepop stack:", stack)
                    u, ee = stack.pop()
                    # print(f"popped {u} {ee}")
                    u, poss_v = stack[-1]
                    v = mapping_G1[u]
                    # print(f"other new v {v}")

                mapping_G2[v] = None
                mapping_G1[u] = None

                for node in G1_neighbors(u):
                    T1[node] -= 1
                    T1_tilda[node] += 1

                for node in G2_neighbors(v):
                    T2[node] -= 1
                    T2_tilda[node] += 1

                T1[u] = T1[u] * (-1)
                T1_tilda[u] += 1
                T2[v] = T2[v] * (-1)
                T2_tilda[v] += 1
                # print(f"Done with ({u}, {v}). Backing out")

        return True

    def is_isomorphic_recursive(self, depth=0):
        """Returns True if the two graphs are isomorphic

        Parameter
        ---------
        depth : int (default=0)
            Index in G1_node_order of the node to match
            This should be 0 to start the process. Later
            recursive calls use other values.

        Examples
        --------
            >>> G1 = nx.path_graph(3)
            >>> nx.set_node_attributes(G1, "red", "color")
            >>> G2 = nx.path_graph(range(7,10))
            >>> nx.set_node_attributes(G2, "red", "color")
            >>> GM = GraphMatcher_vf2pp(G1, G2, "color")
            >>> GM.is_isomorphic()
            True
        """
        # Is the mapping complete?
        if depth == len(self.G1):
            return True

        # Get the node we are currently trying to match
        u = self.G1_node_order[depth]

        # Get the candidate pairs for this node
        poss_v = self._cand_pairs(u)

        # Remove candidates with wrong degree
        u_degree = self.G1.degree(u)
        G2_degree = dict(self.G2.degree)
        poss_v = {v for v in poss_v if G2_degree[v] == u_degree}

        # Make sure there is a candidate
        if not poss_v:
            return False

        # Go through the remaining candidates one by one
        for v in poss_v:

            # Check the criteria
            if self._can_be_isomorphism(u, v) and not self._cant_be_isomorphism(u, v):

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
                if self.is_isomorphic_recursive(depth + 1):
                    return True

                # Otherwise unmatch it and revert the sets
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

    def is_induced_subgraph_isomorphic_recursive(self, depth=0):
        """Returns True if G1 is isomorphic to an induced subgraph of G2

        Parameter
        ---------
        depth : int (default=0)
            Index in G1_node_order of the node to match.
            This should be 0 to start the process. Later
            recursive calls use other values.

        Examples
        --------
            >>> G1 = nx.path_graph(3)
            >>> nx.set_node_attributes(G1, "red", "color")
            >>> G2 = nx.path_graph(7)
            >>> nx.set_node_attributes(G2, "red", "color")
            >>> GM = GraphMatcher_vf2pp(G1, G2, "color")
            >>> GM.is_induced_subgraph_isomorphic_recursive()
            True
            >>> G1.add_edge(1, 3)
            >>> G1.nodes[3]["color"] = "red"
            >>> GM = GraphMatcher_vf2pp(G1, G2, "color")
            >>> GM.is_induced_subgraph_isomorphic_recursive()
            False
        """
        # Is the mapping complete?
        if depth == len(self.G1):
            return True

        # Get the node we are currently trying to match
        u = self.G1_node_order[depth]

        # Get the candidate pairs for this node
        poss_v = self._cand_pairs(u)

        # Remove candidates with wrong degree
        u_degree = self.G1.degree(u)
        G2_degree = dict(self.G2.degree)
        poss_v = [
            v
            for v in poss_v
            if G2_degree[v] >= u_degree
            if self._can_be_isomorphism(u, v)
            if not self._cant_be_ind_subg_iso(u, v)
        ]
        # print("CHECKING", u, poss_v)

        # Make sure there is a candidate
        if not poss_v:
            # print("returning false -- no nodes left")
            return False

        # Go through the remaining candidates one by one
        for i, v in enumerate(poss_v):

            # Check the criteria
            if self._can_be_isomorphism(u, v) and not self._cant_be_ind_subg_iso(u, v):

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
                # print(f"Processed ({u}, {v}) and moving to depth {depth + 1}")
                # print(f"mapping: {self.mapping_G1}")
                # print(f"adding to 'stack': ({u}, {poss_v[i:]})")
                if self.is_induced_subgraph_isomorphic_recursive(depth + 1):
                    return True
                # print(f"Backed out: {u},{poss_v[i:]}")

                # Otherwise unmatch it and revert the sets
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

        # print(f"maybe Done with ({u}, {v}). Backing out")
        # If none of the candidate pairs worked then return False
        return False

    def is_subgraph_monomorphic_recursive(self, depth=0):
        """Returns True if G1 is monomorphic to a subgraph of G2

        Monomorphic means isomorphic to a non-induced subgraph.
        That is, isomorphic to a subset of nodes and edges of G2.

        Parameter
        ---------
        depth : int (default=0)
            Index in G1_node_order of the node to match next.
            This should be 0 to start the process.
            Later recursive calls use other values.

        Examples
        --------
            >>> G1 = nx.path_graph(3)
            >>> nx.set_node_attributes(G1, "red", "color")
            >>> G2 = nx.cycle_graph(3)
            >>> nx.set_node_attributes(G2, "red", "color")
            >>> GM = GraphMatcher_vf2pp(G1, G2, "color")
            >>> GM.is_subgraph_monomorphic_recursive()
            True
            >>> GM = GraphMatcher_vf2pp(G2, G1, "color")
            >>> GM.is_subgraph_monomorphic_recursive()
            False
        """
        # Is the mapping complete?
        if depth == len(self.G1):
            return True

        # Get the node we are currently trying to match
        u = self.G1_node_order[depth]

        # Get the candidate pairs for this node
        poss_v = self._cand_pairs(u)

        # Remove candidates with wrong degree
        u_degree = self.G1.degree(u)
        G2_degree = dict(self.G2.degree)
        poss_v = {v for v in poss_v if G2_degree[v] >= u_degree}

        # Make sure there is a candidate
        if poss_v == []:
            return False

        # Go through the remaining candidates one by one
        for v in poss_v:
            # Check the criteria
            if self._can_be_monomorphism(u, v) and not self._cant_be_monomorphism(u, v):

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
                # print(f"Processed ({u}, {v}) and recursing")
                if self.is_subgraph_monomorphic_recursive(depth + 1):
                    return True

                # Otherwise unmatch it and revert the sets
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
                # print(f"Done with ({u}, {v}). Backing out!")

        # If none of the candidate pairs worked then return False
        return False


def is_isomorphic(G1, G2, label):
    GM = GraphMatcher_vf2pp(G1, G2, label)
    return GM.is_isomorphic()


def is_isomorphic_recursive(G1, G2, label):
    GM = GraphMatcher_vf2pp(G1, G2, label)
    return GM.is_isomorphic_recursive()


def is_induced_subgraph_isomorphic(G1, G2, label):
    GM = GraphMatcher_vf2pp(G1, G2, label)
    return GM.is_induced_subgraph_isomorphic()


def is_induced_subgraph_isomorphic_recursive(G1, G2, label):
    GM = GraphMatcher_vf2pp(G1, G2, label)
    return GM.is_induced_subgraph_isomorphic_recursive()


def is_subgraph_monomorphic(G1, G2, label):
    GM = GraphMatcher_vf2pp(G1, G2, label)
    return GM.is_subgraph_monomorphic()


def is_subgraph_monomorphic_recursive(G1, G2, label):
    GM = GraphMatcher_vf2pp(G1, G2, label)
    return GM.is_subgraph_monomorphic_recursive()
