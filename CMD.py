class CMD:
  def __init__(self):
    self.filename = ""
    self.map_t=0.00;
    self.Nthr=2;
    self.dreso=16.00;
    self.MergeDist=0.50;
    self.Filter=0.10;
    self.Mode=0;
    self.LocalR=10.0;
    self.Dkeep=0.5;
    self.Nround=5000;
    self.Nnb=30;
    self.Ntabu=100;
    self.Nsim=10;
    self.Allow=1.01;
    self.Nbeam=20;
    self.ssize=7.0;
    self.ang=30.0;
    self.TopN=10;
    self.ShowGrid=False;
    self.th1=0.00;
    self.th2=0.00;
    self.Mode=1;
    self.Emode=False;
    self.file1="Data\emd_8097.mrc"
    self.file2="Data\ChainA_simulated_resample.mrc"

class mrc_obj:
    def __init__(self, path):
        mrc = mrcfile.open(path)
        data = mrc.data
        header = mrc.header
        self.xdim = int(header.nx)
        self.ydim = int(header.ny)
        self.zdim = int(header.nz)
        self.xwidth = mrc.voxel_size.x
        self.ywidth = mrc.voxel_size.y
        self.zwidth = mrc.voxel_size.z
        self.cent = [
            self.xdim * 0.5,
            self.ydim * 0.5,
            self.zdim * 0.5,
        ]
        self.orig = {"x": header.origin.x, "y": header.origin.y, "z": header.origin.z}
        self.data = copy.deepcopy(data)
        self.dens = data.flatten()
        self.vec = np.zeros((self.xdim * self.ydim * self.zdim, 3))
        self.dsum = None
        self.Nact = None
        self.ave = None
        self.std_norm_ave = None
        self.std = None