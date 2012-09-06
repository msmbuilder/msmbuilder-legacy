import os
import inspect

def fixtures_dir():
    #http://stackoverflow.com/questions/50499/in-python-how-do-i-get-the-path-and-name-of-the-file-that-is-currently-executin
    return os.path.join(os.path.dirname(inspect.getfile(inspect.currentframe())), 'fixtures')
