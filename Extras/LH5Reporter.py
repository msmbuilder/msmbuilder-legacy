"""
LH5reporter.py: Output a trajectory in MSMBuilder LH5 format.

This class is for use with the Python OpenMM App.  This class allows OpenMM to directly write an MSMBuilder2 LH5 Trajectory.

Example: After preparing a simulation in OpenMM, the following code allows OpenMM to directly write to a *h5 file.

R1=Trajectory.Trajectory.LoadFromPDB('native.pdb')
simulation.reporters.append(LH5Reporter.LH5Reporter('Trajectories/trj0.lh5', 2000,R1))
"""

import simtk.openmm as mm
import simtk.unit as units
from msmbuilder import Trajectory
import tables
    
class LH5Reporter(object):
    """LH5Reporter outputs a series of frames from a Simulation to a PDB file.
    
    To use it, create a LH5Reporter, than add it to the Simulation's list of reporters.
    """
    
    def __init__(self, file, reportInterval,TrajObject):
        """Create a LH5Reporter.
    
        Parameters:
         - file (string) The file to write to
         - reportInterval (int) The interval (in time steps) at which to write frames
        """
        self._reportInterval = reportInterval
        self._topology = None
        self._nextModel = 0
        self.TrajObject=TrajObject
        self.NumAtoms=self.TrajObject.GetNumberOfAtoms()
        self.PrepareH5File(file)

        
    def PrepareH5File(self,filename):
        
        self.TrajObject.SaveToLHDF(filename)
        self.HDFFile=tables.File(filename,'a')
        self.HDFFile.removeNode("/","XYZList")
        
        self.HDFFile.createEArray("/","XYZList",tables.Float32Atom(),shape=[0,self.NumAtoms,3],filters=Trajectory.Serializer.Filter)
        self.HDFFile.flush()
        
    
    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.
        
        Parameters:
         - simulation (Simulation) The Simulation to generate a report for
        Returns: A five element tuple.  The first element is the number of steps until the
        next report.  The remaining elements specify whether that report will require
        positions, velocities, forces, and energies respectively.
        """
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False)
    
    def report(self, simulation, state):
        """Generate a report.
        
        Parameters:
         - simulation (Simulation) The Simulation to generate a report for
         - state (State) The current state of the simulation
        """
        xyz=Trajectory.ConvertToLossyIntegers(state.getPositions(asNumpy=True).value_in_unit(units.nanometers),1000)
        
        self.HDFFile.root.XYZList.append([xyz])
        self.HDFFile.flush()
        self._nextModel += 1
    
    def __del__(self):
        self.HDFFile.close()
        self._out.close()
