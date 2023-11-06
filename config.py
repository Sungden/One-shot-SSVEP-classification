import os
from datetime import datetime
class Config(object):
    def __init__(self):
        
        self.CNN='DNN_LST' # DNN, DNN_LST or MyNet
        #####need to be various
        self.Dataset=2#0 for UCSD, 1 for benchmark, 2 for BETA
        self.sample_length=0.5

        self.patience=300

        
        # ### UCSD dataset
        # self.C = 8 
        # self.rfs=256
        # self.dropout=0.95
        # self.num_class = 12
        
        ## BETA and Bench dataset
        self.C = 9 
        self.rfs=250
        self.dropout=0.95  
        self.num_class = 40

        self.T = int(self.sample_length*self.rfs)#500
        self.lr = 2*1e-4
        self.val = 36       
        self.batchsize = 32
        self.epoch = 100
        self.Nh=4
        self.Nm=3
        self.smooth = 0.01

        
        self.save_path = os.path.abspath(os.curdir)+'/result/'
            
        
config = Config()
