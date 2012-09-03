import os

def remount():

    mount_in_path = False
    unmount_in_path = False
    for p in os.environ['PATH'].split(':'):
        if os.path.exists( os.path.join( p, 'mountSSHFS.sh') ):
            mount_in_path=True
        if os.path.exists( os.path.join( p, 'unmountSSHFS.sh' ) ):
            unmount_in_path=True

    if (not mount_in_path) or (not unmount_in_path):
        print "Cannot find mountSSHFS.sh or unmountSSHFS.sh in PATH... Not re-mounting"
        return
        

    cwd = os.getcwd()
    
    os.chdir( os.environ['HOME'] )
    os.system('unmountSSHFS.sh')
    os.system('mountSSHFS.sh')
    os.chdir( cwd )

    return
