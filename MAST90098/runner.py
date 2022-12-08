import numpy as np
from numpy.core.numeric import cross
from numpy.lib.npyio import save
import pandas as pd
from time import time
import random
import os
from tqdm import tqdm


from GA import GeneticMakespan

def random_jobs(n, LB=1, UB=101):
    p = np.random.randint(LB, UB, size=n)
    return p


def generate_instances(nums=10, save=False):
    N = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 10000]
    M = [[(k*n)//10 for k in range(1,10)] for n in N ]
    
    gl_stack = []
    for i in range(len(M)):
        
        for j in range(len(M[i])):
            for k in range(nums):
                instance = [N[i], M[i][j]] + random_jobs(N[i]).tolist()
                name = 'instance_n='+str(N[i])+'_m='+str(M[i][j])+'_'+str(k+1)+'.txt'
                if save:
                    out_dir = './data'
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    

                    np.savetxt(os.path.join(out_dir, name), instance,fmt='%d', delimiter='\n') 

                    
        gl_stack.append(instance)
    
    if not save:
        return gl_stack


def load_instances(filename, in_dir='./data'):
    instances = np.loadtxt(os.path.join(in_dir, filename), delimiter='\n').astype(int)
    return instances


def collect_generated(runs=10, hybrid=False):
    if not os.path.exists('./data'):
        generate_instances(save=True)

    
    N = [200, 300, 400, 500, 600, 700, 800, 900]
    M = [[(k*n)//10 for k in range(1,10)] for n in N ]
    solns = []
    for i in range(len(M)):
        for j in range(len(M[i])):
            times = 0
            run_record = ['n='+str(N[i])+'_m='+str(M[i][j])]
            for k in tqdm(range(runs)):
                filename = 'instance_n='+str(N[i])+'_m='+str(M[i][j])+'_'+str(k+1)+'.txt'
                instance = load_instances(filename=filename)
                solver = GeneticMakespan(
                    pop_size=100, 
                    parent_num=20, 
                    max_gen=100,
                    time_limit=2*60.0,
                    hybrid=hybrid
                )
                T_best, C_best, runtime = solver.solve((instance[0], instance[1], instance[2:]))
                run_record.append(max(C_best))
                times += runtime
            run_record.insert(1, times/runs)
            solns.append(run_record)
            
    df = pd.DataFrame(solns,columns=None,index=None)

    out_dir = './results'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    filename = 'all_runs_ga.csv'
    if hybrid:
        filename = 'all_runs_hga_new.csv'
    
    
    df.to_csv(os.path.join(out_dir, filename), index=False,header=False)
            

            
def main():
    generate_instances(nums=10)
    collect_generated(hybrid=False)

if __name__ == '__main__':
    main()
    
    
    