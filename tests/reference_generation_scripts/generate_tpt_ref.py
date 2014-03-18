from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import datetime

import numpy as np
import scipy.sparse
from scipy import io

from msmbuilder import transition_path_theory as tpt
from msmbuilder import Serializer

d = datetime.date.today()
BACKUP_DIR = 'tpt_ref_%s.%s.%s' % (d.month, d.day, d.year)
os.mkdir(BACKUP_DIR)

def save_new_ref(filename, data):
    """ Saves a new version of the reference data, and backs up the old """
    
    ext = filename.split('.')[-1]
    
    if (data == None):
        print("WARNING: Error generating file: %s" % filename)
        print("Skipped... try again.")
        return
    
    if os.path.exists(filename):
        os.system( 'mv %s %s' % (filename, BACKUP_DIR) )
    
    if ext in ['h5', 'lh5']:
        if scipy.sparse.issparse(data):
            data = data.toarray()
        Serializer.SaveData(filename, data)
    elif ext == 'mtx':
        io.mmwrite(filename, data)
    elif ext == 'pkl':
        f = open(filename, 'w')
        pickle.dump(f, data)
        f.close()
    else:
        raise ValueError('Could not understand extension (.%s) for %s' % (ext, filename))
    
    return


# hard-coding these right now - would be better to read them from a common
# place, where the actual unit test also gets these values

tpt_ref_dir = '.'
tprob = io.mmread( os.path.join( "tProb.mtx") ) #.toarray() # strange bug??
sources   = [0]   # chosen arbitarily by TJL
sinks     = [70]  # chosen arbitarily by TJL
waypoints = [60]  # chosen arbitarily by TJL
lag_time  = 1.0   # chosen arbitarily by TJL

# committors
Q = tpt.calculate_committors(sources, sinks, tprob)
save_new_ref('committors.h5', Q)

# flux
flux = tpt.calculate_fluxes(sources, sinks, tprob)
save_new_ref('flux.h5', flux)

net_flux = tpt.calculate_net_fluxes(sources, sinks, tprob)        
save_new_ref('net_flux.h5', net_flux)        
        
# path calculations
paths, bottlenecks, fluxes = tpt.find_top_paths(sources, sinks, tprob)
save_new_ref('dijkstra_paths.h5', paths)
save_new_ref('dijkstra_fluxes.h5', fluxes)
save_new_ref('dijkstra_bottlenecks.h5', bottlenecks)

# MFPTs
mfpt = tpt.calculate_mfpt(sinks, tprob, lag_time=lag_time)
save_new_ref('mfpt.h5', mfpt)

ensemble_mfpt = tpt.calculate_ensemble_mfpt(sources, sinks, tprob, lag_time)
save_new_ref('ensemble_mfpt.h5', ensemble_mfpt)

all_to_all_mfpt = tpt.calculate_all_to_all_mfpt(tprob)
save_new_ref('all_to_all_mfpt.h5', all_to_all_mfpt)

# TP Time
tp_time = tpt.calculate_avg_TP_time(sources, sinks, tprob, lag_time)
save_new_ref('tp_time.h5', tp_time)
   
# hub scores
#frac_visits = tpt.calculate_fraction_visits(tprob, waypoints, 
#                                            sources, sinks)
#save_new_ref('frac_visits.h5', frac_visits)
        
#hub_score = tpt.calculate_hub_score(tprob, waypoints)
#save_new_ref('hub_score.h5', hub_score)
        
#all_hub_scores = tpt.calculate_all_hub_scores(tprob)
#save_new_ref('all_hub_scores.h5', all_hub_scores)

# done!
print("Done generating new reference for transition_path_theory.py!")


