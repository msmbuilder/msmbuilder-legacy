from __future__ import print_function, absolute_import, division
from mdtraj.utils.six import PY2

import numpy as np
import itertools
from numbers import Number
import mdtraj as md

from .baseclasses import Vectorized, AbstractDistanceMetric


class ContinuousContact(Vectorized, AbstractDistanceMetric):

    """Distance metric for calculating distances between frames based on the
    pairwise distances between residues.

    Here each frame is represented as a vector of the distances between pairs
    of residues.
    """

    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'seuclidean', 'mahalanobis']

    def __init__(self, metric='euclidean', p=2, contacts='all', scheme='closest-heavy',
                 V=None, VI=None):
        """Create a distance calculator based on the distances between pairs of atoms
        in a sturcture -- like the contact map except without casting to boolean.

        Parameters
        ----------
        metric : {'braycurtis', 'canberra', 'chebyshev', 'cityblock',
                  'correlation', 'cosine', 'euclidean', 'minkowski',
                  'sqeuclidean'}
            distance metric to equip the space with
        p : int
            exponent for p-norm, used only for `metric='minkowski'`
        contacts : {ndarray, 'all'}
            contacts can be an n by 2 array, where each row is a pair of
            integers giving the indices of 2 residues whose distance you care about.
            Alternatively, contacts can be the string 'all'. This is a shortcut for
            supplying a contacts list that includes all (N-2 * N-3) / 2 pairs of
            residues which are more than 2 residues apart.
        scheme: {'CA', 'closest', 'closest-heavy'}
            scheme can be 'CA', 'closest', or 'closest-heavy' and gives
            the sense in which the 'distance between two residues' is computed. If
            scheme is 'CA', then we'll use the cartesian distance between the residues'
            C-alpha atoms as their distance for the purpose of calculating whether or
            not they have exceeded the cutoff. If scheme is 'closest', we'll use the
            distance between the closest pair of atoms where one
            belongs to residue i and to residue j. If scheme is 'closest-heavy', we'll
            use the distance between the closest pair of non-hydrogen atoms where one
            belongs to reside i and one to residue j.
        """
        s = super(ContinuousContact, self) if PY2 else super()
        s.__init__(metric, p, V, VI)

        self.contacts = contacts

        scheme = scheme.lower()
        if not scheme in ['ca', 'closest', 'closest-heavy']:
            raise ValueError('Unrecognized scheme')

        self.scheme = scheme

    def __repr__(self):
        try:
            contacts_repr = repr(self.contacts.tolist())
        except:
            contacts_repr = repr(self.contacts)
        return 'metrics.ContinuousContact(metric=%s, p=%s, contacts=%s, scheme=%s)' % (self.metric, self.p, contacts_repr, self.scheme)

    def prepare_trajectory(self, trajectory):
        """Prepare a trajectory for distance calculations based on the contact map.

        Each frame in the trajectory will be represented by a vector where
        each entries represents the distance between two residues in the structure.
        Depending on what contacts you pick to use, this can be a 'native biased'
        picture or not.

        Paramters
        ---------
        trajectory : mdtraj.Trajectory
            The trajectory to prepare

        Returns
        -------
        pairwise_distances : ndarray
            1D array of various residue-residue distances
        """
        return md.compute_contacts(trajectory, self.contacts, self.scheme)


class BooleanContact(Vectorized, AbstractDistanceMetric):

    """Distance metric for calculating distances between frames based on their
    contact maps.

    Here each frame is represented as a vector of booleans representing whether
    the distance between pairs of residues is less than a cutoff.
    """

    allowable_scipy_metrics = ['dice', 'kulsinki', 'matching', 'rogerstanimoto',
                               'russellrao', 'sokalmichener', 'sokalsneath',
                               'yule']

    def __init__(self, metric='matching', contacts='all', cutoff=0.5, scheme='closest-heavy'):
        """ Create a distance metric that will measure the distance between frames
        based on differences in their contact maps.

        Paramters
        ---------
        metric : {'dice', 'kulsinki', 'matching', 'rogerstanimoto',
                  russellrao', 'sokalmichener', 'sokalsneath', 'yule'}
            You should probably use matching. Then the distance between two
            frames is just the number of elements in their contact map that are
            the same. See the scipy.spatial.distance documentation for details.
        contacts : {ndarray, 'all'}
            contacts can be an n by 2 array, where each row is a pair of
            integers giving the indices of 2 residues which form a native contact.
            Each conformation is then represnted by a vector of booleans representing
            whether or not that contact is present in the conformation. The distance
            metric acts on two conformations and compares their vectors of booleans.
            Alternatively, contacts can be the string 'all'. This is a shortcut for
            supplying a contacts list that includes all (N-2 * N-3) / 2 pairs of
            residues which are more than 2 residues apart.
        cutoff : {float, ndarray}
            cutoff can be either a positive float representing the cutoff distance between two
            residues which constitues them being 'in contact' vs 'not in contact'. It
            is measured in the same distance units that your trajectory's XYZ data is in
            (probably nanometers).
            Alternatively, cutoff can be an array of length equal to the number of rows in the
            contacts array, specifying a different cutoff for each contact. That is, cutoff[i]
            should contain the cutoff for the contact in contact[i].
        scheme : {'CA', 'closest', 'closest-heavy'}
            scheme can be 'CA', 'closest', or 'closest-heavy' and gives
            the sense in which the 'distance between two residues' is computed. If
            scheme is 'CA', then we'll use the cartesian distance between the residues'
            C-alpha atoms as their distance for the purpose of calculating whether or
            not they have exceeded the cutoff. If scheme is 'closest', we'll use the
            distance between the closest pair of atoms where one belongs to residue i
            and to residue j. If scheme is 'closest-heavy', we'll use the distance
            between the closest pair of non-hydrogen atoms where one belongs to reside
            i and one to residue j."""

        super(BooleanContact, self).__init__(metric)
        self.contacts = contacts

        if isinstance(cutoff, Number):
            self.cutoff = cutoff
        else:
            self.cutoff = np.array(cutoff).flatten()

        scheme = scheme.lower()
        if not scheme in ['ca', 'closest', 'closest-heavy']:
            raise ValueError('Unrecognized scheme')

        self.scheme = scheme

    def __repr__(self):
        try:
            contacts_repr = repr(self.contacts.tolist())
        except:
            contacts_repr = repr(self.contacts)

        try:
            cutoff_repr = repr(self.cutoff.tolist())
        except:
            cutoff_repr = repr(self.cutoff)

        return 'metrics.BooleanContact(metric=%s, p=%s, contacts=%s, cutoff=%s, scheme=%s)' % (self.metric, self.p, contacts_repr, cutoff_repr, self.scheme)

    def prepare_trajectory(self, trajectory):
        """Prepare a trajectory for distance calculations based on the contact map.

        Paramters
        ---------
        trajectory : mdtraj.Trajectory
            The trajectory to prepare

        Returns
        -------
        pairwise_distances : ndarray
            1D array of various residue-residue distances, casted to boolean
        """

        ccm = ContinuousContact(contacts=self.contacts, scheme=self.scheme)
        contact_d = ccm.prepare_trajectory(trajectory)
        if not isinstance(self.cutoff, Number):
            # contact_d has frames in rows and contacts in columns
            if not len(self.cutoff) == contact_d.shape[1]:
                raise ValueError('cutoff must be a number or match the length of contacts')

        # contact = np.zeros_like(contact_d).astype(bool)
        # for i in xrange(contact_d.shape[0]):
        #    contact[i, :] = contact_d[i, :] < self.cutoff
        contact = contact_d < self.cutoff
        return contact


class AtomPairs(Vectorized, AbstractDistanceMetric):

    """Concrete distance metric that monitors the distance
    between certain pairs of atoms (as opposed to certain pairs of residues
    as ContinuousContact does)"""

    allowable_scipy_metrics = ['braycurtis', 'canberra', 'chebyshev', 'cityblock',
                               'correlation', 'cosine', 'euclidean', 'minkowski',
                               'sqeuclidean', 'seuclidean', 'mahalanobis']

    def __init__(self, metric='cityblock', p=1, atom_pairs=None, V=None, VI=None):
        """ Atom pairs should be a N x 2 array of the N pairs of atoms
        whose distance you want to monitor"""
        super(AtomPairs, self).__init__(metric, p, V=V, VI=VI)
        try:
            atom_pairs = np.array(atom_pairs, dtype=int)
            n, m = atom_pairs.shape
            if not m == 2:
                raise ValueError()
        except (ValueError, TypeError):
            raise ValueError('Atom pairs must be an n x 2 array of pairs of atoms')
        self.atom_pairs = np.int32(atom_pairs)

    def prepare_trajectory(self, trajectory):
        return md.compute_distances(trajectory, self.atom_pairs)
