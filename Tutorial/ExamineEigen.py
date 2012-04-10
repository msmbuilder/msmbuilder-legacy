from Emsmbuilder import Serializer, MSMLib, Conformation, Trajectory, lumping
import scipy.io

MicroDir = "./Data/"
Unit = "ps"
System = "Alanine Dipeptide"
LagTime = 3.0
max_eigen = 20

T=scipy.io.mmread(MicroDir+"/tProb.mtx").toarray()
N=max_eigen
l,v=MSMLib.GetEigenvectors(T,N)

phi = Serializer.LoadData("./Phi.h5")
psi = Serializer.LoadData("./Psi.h5")

Ass = Serializer.LoadData(MicroDir+"Assignments.h5")

num_states = Ass.max()+1
figure()
a = Ass.copy()
mapping = zeros(num_states,'int')
mapping[v[:,1] > 0] = 1
MSMLib.ApplyMappingToAssignments(a,mapping)
w=lambda x: where(a==x)
plot(phi[w(0)],psi[w(0)],"x")
plot(phi[w(1)],psi[w(1)],"+")

title("Ramachandran plot of alanine dipeptide MSM.")
xlabel(r"$\phi$")
ylabel(r"$\psi$")
plot([-180,0],[50,50],'k')
plot([-180,0],[-100,-100],'k')
plot([0,180], [100,100],'k')
plot([0,180], [-50,-50],'k')
plot([0,0],[-180,180],'k')
plot([-100,-100],[50,180],'k')
plot([-100,-100],[-180,-100],'k')

axis([-180,180,-180,180])
