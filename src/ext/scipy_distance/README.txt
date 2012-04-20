This is a modified version of scipy.spatial.distance with
openMP parallel pragmas to the cdist methods


From python, the c module is called _distance_wrap
i.e. >> from euclid import _distance_wrap.