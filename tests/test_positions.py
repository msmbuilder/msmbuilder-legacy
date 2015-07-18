from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from msmbuilder.metrics import Positions
from msmbuilder.testing import get, eq, expected_failure
import numpy as np
import unittest

class TestPositions():

    def setup(self):
        self.ala = get('native.pdb')
        self.rotZ = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        self.rotated_ala = get('native.pdb')
        self.rotated_ala.xyz[0] = np.array([self.rotZ.dot(atom) for atom in self.ala.xyz[0]])

    @expected_failure
    def test_negative(self):
        eq(self.ala.xyz, self.rotated_ala.xyz)

    def test(self):
        pos_metric = Positions(self.ala)
        protated_ala = pos_metric.prepare_trajectory(self.rotated_ala)

        eq(self.ala.xyz.reshape((1, -1)), protated_ala)
