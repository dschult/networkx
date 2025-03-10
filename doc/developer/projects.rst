Mentored Projects
==================

This page maintains a list of mentored project ideas that contributors can work
on if they are interested in contributing to the NetworkX project. Feel free to
suggest any other idea if you are interested on the
`NetworkX GitHub discussions page <https://github.com/networkx/networkx/discussions>`__

These ideas can be used as projects for Google Summer of Code, Outreachy,
NumFOCUS Small Development Grants and university course/project credits (if
your university allows contribution to open source for credit).

If you are a professor interested in having your class develop tools
and contribute the results to NetworkX, we welcome your submissions!
We encourage you to consider rewriting/improving existing functions
as a potential source of projects. Working off of and
improving existing tools involves reading, evaluating and writing code,
rather than just writing new code. We also encourage them to review each
other's PRs. You can have students submit their PRs to your personal fork,
discuss, review, etc in an environment conducive to mentoring and learning.
Once their branch is ready to merge, it can be submitted to the main NetworkX
repository. This will help keep in-class discussions separate from the
broader NetworkX review process, making both more manageable and readable.
Feel free to reach out to use as you plan these activities.

Pedagogical Interactive Notebooks for Algorithms Implemented in NetworkX
------------------------------------------------------------------------

- Abstract: NetworkX has a :ref:`wide variety of algorithms <Algorithms>`
  implemented. Even though the algorithms are well documented, explanations of
  the ideas behind the algorithms are often missing and we would like to
  collect these, write Jupyter notebooks to elucidate these ideas and explore
  the algorithms experimentally, and publish the notebooks at
  https://github.com/networkx/notebooks. The goal is to gives readers a
  deeper outlook behind standard network science and graph theory algorithms
  and encourage them to delve further into the topic.

- Recommended Skills: Python, Jupyter notebooks, graph algorithms.

- Expected Outcome: A collection of Interactive Jupyter notebooks which
  explain and explore network algorithms to readers and users of NetworkX.
  For example, see this notebook on
  :doc:`Geometric Generator Models <nx-guides:content/generators/geometric>`

- Complexity: Depending on the algorithms you are interested to work on.

- Interested Mentors: `@MridulS <https://github.com/MridulS/>`__,
  `@rossbar <https://github.com/rossbar/>`__

- Expected time commitment: This project can be either a medium project (~175 hours)
  or a large project (~350 hours). The contributor is expected to contribute 2-3
  pedagogical interactive notebooks for the medium duration project and 4-5 notebooks
  for the long duration project.

Incorporate a Python library for ISMAGs isomorphism calculations
----------------------------------------------------------------

- Abstract: A team from Sandia Labs has converted the original java implementation of
  the ISMAGS isomorphism routines to Python. They have invited us to incorporate that
  code into NetworkX if we are interested. We'd like someone to learn the ISMAGS code
  we currently provide, and the code from this new library and figure out what the
  best combination is to include in NetworkX moving forward. That could be two separate
  subpackages of tools, or more likely a combination of the two sets of code, or a
  third incantation that combines good features from each.

- Recommended Skills: Python, graph algorithms.

- Expected Outcome: A plan for how to best incorporate ISMAGS into NetworkX along
  with code to do that incorporation.

- Interested Mentors: `@dschult <https://github.com/dschult/>`__,
  `@rossbar <https://github.com/rossbar/>`__

- Expected time commitment: This project will be a full time 10 week project (~350 hrs).

Centrality Atlas
----------------

- Abstract: The goal of this project would be to produce a comprehensive review
  of network centrality measures.
  Centrality is a central concept in network science and has many applications
  across domains. NetworkX provides many functions for measuring
  various types of :doc:`network centrality</reference/algorithms/centrality>`.
  The individual centrality functions are typically well-described by their
  docstrings (though there's always room for improvement!); however, there
  currently is no big-picture overview of centrality.
  Furthermore, many of the centrality measures are closely related, but there is
  no documentation that describes these relationships.

- Recommended Skills: Python, literature review, technical writing

- Expected Outcome: An executable document that provides an overview and applications
  of network centrality measures. Potential outputs include (but are not limited
  to): an article for ``nx-guides`` (see above) and/or an example gallery for centrality
  measures.

- Interested Mentors: `@dschult <https://github.com/dschult/>`__,
  `@rossbar <https://github.com/rossbar/>`__

- Expected time commitment: Variable, though a high-quality review article would
  be expected to take several months of dedicated research (~350 hours).

Completed Projects
==================

- `Revisiting and expanding nx-parallel`_
    - Program: Google Summer of Code 2024
    - Contributor: `@Schefflera-Arboricola <https://github.com/Schefflera-Arboricola>`__
    - Link to Proposal: `GSoC 2024: Revisiting and expanding nx-parallel <https://github.com/networkx/archive/blob/main/proposals-gsoc/GSoC-2024-Revisiting-and-expanding-nx-parallel.pdf>`_

- `Unifying the Visualization Interface for NetworkX`
    - Program: Funded by a grant from CZI (Chan-Zuckerberg Initiative) 2024
    - Contributor: `@mjschwenne <https://github.com/mjschwenne>`__
    - Summary: Refactored existing draw functions to a single function with all drawing
      attributes stored in the NetworkX Graph object.

- `VF2++ algorithm for graph isomorphism`_
    - Program: Google Summer of Code 2022
    - Contributor: `@kpetridis24 <https://github.com/kpetridis24/>`__
    - Link to Proposal: `GSoC 2022: VF2++ Algorithm <https://github.com/networkx/archive/blob/main/proposals-gsoc/GSoC-2022-VF2plusplus-isomorphism.pdf>`_

- `Louvain community detection algorithm`_
    - Program: Google Summer of Code 2021
    - Contributor: `@z3y50n <https://github.com/z3y50n/>`__
    - Link to Proposal:  `GSoC 2021: Community Detection Algorithms <https://github.com/networkx/archive/blob/main/proposals-gsoc/GSoC-2021-Community-Detection-Algorithms.pdf>`__

- `Asadpour algorithm for directed travelling salesman problem`_
    - Program: Google Summer of Code 2021
    - Contributor: `@mjschwenne <https://github.com/mjschwenne/>`__
    - Link to Proposal:  `GSoC 2021: Asadpour algorithm <https://github.com/networkx/archive/blob/main/proposals-gsoc/GSoC-2021-Asadpour-Asymmetric-Traveling%20Salesman-Problem.pdf>`__

- Pedagogical notebook: `Directed acyclic graphs and topological sort`_
    - Program: Google Summer of Code 2021
    - Contributor:  `@vdshk <https://github.com/vdshk>`__

- Pedagogical notebooks: `Graph assortativity`_ & `Network flow analysis and Dinitz algorithm`_
    - Program: Google Summer of Code 2021
    - Contributor: `@harshal-dupare <https://github.com/harshal-dupare/>`__

- Add On system for NetworkX: `NetworkX-Metis`_
    - Program: Google Summer of Code 2015
    - Contributor: `@OrkoHunter <https://github.com/OrkoHunter/>`__
    - Link to Proposal:  `GSoC 2015: Add On System for NetworkX <https://github.com/networkx/archive/blob/main/proposals-gsoc/GSoC-2015-Add-on-system-for-NetworkX.md>`__

- `NetworkX 2.0 API`_
    - Program: Google Summer of Code 2015
    - Contributor: `@MridulS <https://github.com/MridulS/>`__
    - Link to Proposal: `GSoC 2015: NetworkX 2.0 API <https://github.com/networkx/archive/blob/main/proposals-gsoc/GSoC-2015-NetworkX-2.0-api.md>`__

.. _`Revisiting and expanding nx-parallel`: https://github.com/Schefflera-Arboricola/blogs/tree/main/networkx/GSoC24
.. _`VF2++ algorithm for graph isomorphism`: https://github.com/networkx/networkx/pull/5788
.. _`Louvain community detection algorithm`: https://github.com/networkx/networkx/pull/4929
.. _`Asadpour algorithm for directed travelling salesman problem`: https://github.com/networkx/networkx/pull/4740
.. _`Directed acyclic graphs and topological sort`: https://github.com/networkx/nx-guides/pull/44
.. _`Graph assortativity`: https://github.com/networkx/nx-guides/pull/42
.. _`Network flow analysis and Dinitz algorithm`: https://github.com/networkx/nx-guides/pull/46
.. _`NetworkX-Metis`: https://github.com/networkx/networkx-metis
.. _`NetworkX 2.0 API`: https://networkx.org/documentation/latest/release/migration_guide_from_1.x_to_2.0.html

..
   Project Idea Template
   ---------------------

   - Abstract:

   - Recommended Skills:

   - Expected Outcome:

   - Complexity;

   - Interested Mentors:

