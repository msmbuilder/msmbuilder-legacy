Maximum-Likelihood Reversible Transition Matrix
===============================================

Here, we sketch out the objective function and gradient used to
find the maximum likelihood reversible count matrix.

Let :math:`C_{ij}` be the matrix of observed counts. :math:`C` must be
strongly connected for this approach to work! Below, :math:`f` is the
log likelihood of the observed counts.

.. math:: f = \sum_{ij} C_{ij} \log T_{ij}

Let :math:`T_{ij} = \frac{X_{ij}}{\sum_j X_{ij}}`,
:math:`X_{ij} =  \exp(u_{ij})`, :math:`q_i = \sum_j \exp(u_{ij})`

Here, :math:`u_{ij}` is the log-space representation of :math:`X_{ij}`.
It follows that :math:`T_{ij} = \exp(u_{ij}) \frac{1}{q_i}`, so
:math:`\log(T_{ij}) = u_{ij} - \log(q_{i})`

.. math:: f = \sum_{ij} C_{ij} u_{ij} - \sum_{ij} C_{ij} \log q_i

Let :math:`N_i = \sum_j C_{ij}`

.. math:: f = \sum_{ij} C_{ij} u_{ij} - \sum_{i} N_i \log q_i

Let :math:`u_{ij} = u_{ji}` for :math:`i > j`, eliminating terms with
:math:`i>j`.

Let :math:`S_{ij} = C_{ij} + C_{ji}`

.. math:: f = \sum_{i \le j} S_{ij} u_{ij} - \frac{1}{2} \sum_i S_{ii} u_{ii} - \sum_i N_i \log q_i

.. math:: \frac{df}{du_{ab}} = S_{ab}  - \frac{1}{2} S_{ab} \delta_{ab} - \sum_i \frac{N_i}{q_i} \frac{dq_i}{du_{ab}}

.. math:: \frac{dq_i}{du_{ab}} = \exp(u_{ab}) [\delta_{ai} + \delta_{bi} - \delta_{ab} \delta_{ia}]

Let :math:`v_i = \frac{N_i}{q_i}`

.. math:: \sum_i V_i \frac{dq_i}{du_{ab}} = \exp(u_{ab}) (v_a + v_b - v_a \delta_{ab})

Thus,

.. math:: \frac{df}{du_{ab}} = S_{ab} - \frac{1}{2} S_{ab} \delta_{ab} - \exp(u_{ab}) (v_a + v_b - v_a \delta_{ab})


