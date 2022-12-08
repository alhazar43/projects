import numpy as np
from time import time
import random

class GreedyMakespan:
    def __init__(self):
        pass
    
    def solve(self, instance):
        t0 = time()
        self.n = instance[0]
        self.m = instance[1]
        self.jobs = instance[2]
        
        
        self.jobs = np.sort(self.jobs)[::-1]
        T = [[i] for i in range(self.m)]
        Time = self.jobs[:self.m]
        
        
        for i in range(self.m, len(self.jobs)): 
            l = np.argmin(Time)
            T[l].append(i)
            Time[l] += self.jobs[i]

        return T, Time, time()-t0



class GeneticMakespan:
    def __init__(
        self, 
        pop_size, 
        parent_num, 
        cross_type=1, 
        cross_points=2, 
        cross_prob=0.5, 
        mutate_prob=0.05, 
        max_gen=100, 
        time_limit=60.0,
        hybrid=False,
        p0=[]
    ):
        
        
        """ 
        
        Key parameters:
        
        pop_size:       integer, size of population to keep at each generation
        parent_num:     integer (default 2), number of parents to perform crossovers 
                        at each generation (ideally 10-20% of population).
        cross_type:     binary (default 1), 0 = uniform_crossover, 1 = k_points_crossover
        cross_points:   integer (default 2), number of points to exchange gene in 
                        a crossover (i.e. k-point-crossover)
        cross_prob      float (default 0.5), probability for gauging uniform crossover
        mutate_prob:    floate (default 0.05), probability that given offspring might mutate
                        at each generation (ideally a probability of 1/n)
        max_gen:        integer (default 100), number of generation to be computed, 
                        unless terminated by time_limit
        time_limit:     float (in seconds, default 60.0), total running time allowance
        hybrid:         boolean (default False), enable HGA for a proposed muation operator
        
        
        Optional parameters:
        
        p0:             array of length pop_size (default empty), initial population         

        """
        

        self.pop_size = pop_size
        self.parent_num = parent_num
        self.cross_type = cross_type
        self.cross_points = cross_points
        self.cross_prob = cross_prob
        self.mutate_prob = mutate_prob
        self.max_gen = max_gen
        self.time_limit = time_limit
        self.p0 = []
        self.hybrid = hybrid
        
        
        

    def init_population(self):
        p0 = [np.random.randint(self.m, size=self.n) for _ in range(self.pop_size)]
        if self.p0:
            return np.array(self.p0)
        else:
            return np.array(p0)
        
        
    def objective(self, solution):
        C = [0]*self.m
        for i, p in enumerate(solution):
            C[p] += self.jobs[i]
        
        return C
    
    @staticmethod
    def exponential(x, alpha, beta):
        return alpha * np.exp(-beta*np.array(x))
    
    
    def k_point_crossover(self, p1, p2, points):
        for point in points:
            t1 = np.concatenate((p1[:point], p2[point:]))
            p2 = np.concatenate((p2[:point], p1[point:]))
            p1 = t1
        return p1, p2
    
    def uniform_crossover(self, p1, p2):
        for i in range(len(p1)):
            if np.random.rand()>self.cross_prob:
                tmp = p1[i].copy()
                p1[i] = p2[i].copy()
                p2[i] = tmp
        return p1, p2
    
    
    def mutation(self, chrom):
        for i in range(len(chrom)):
            if np.random.rand() < self.mutate_prob:
                new_allele = np.random.randint(self.m)
                while chrom[i] == new_allele:
                    new_allele = np.random.randint(self.m)
                chrom[i] = new_allele
        return chrom

    def contorlled_mutation(self, chrom):
        C = self.objective(chrom)
        F = self.exponential(C, alpha=1.0, beta=1/self.n)

        bad_alleles = np.argsort(F)[:self.n//2]
        new_alleles = np.argsort(F)[::-1][:self.n//2]
        n_prob = F[new_alleles]/sum(F[new_alleles])
        for i in range(len(chrom)):
            j = np.random.choice(new_alleles, p=n_prob)

            if chrom[i] in bad_alleles:
                if len(chrom[chrom==chrom[i]])> self.n//self.m:
                    chrom[i] = j
                if len(chrom[chrom==j])> np.ceil(self.n/self.m):
                    new_alleles = np.setdiff1d(new_alleles, [j])
                    n_prob = F[new_alleles]/sum(F[new_alleles])
        return chrom
    
    def solve(self, instance, worst_ms=False):
        t0 = time()
        
        gen_best = []
        gen_worst = []
        

        
        if len(instance) != 3:
            raise ValueError('Cannot recognize the given instance, please reformat and try again')
        
        
        self.n = instance[0]
        self.m = instance[1]
        self.jobs = instance[2]
        
        alpha = 1.0
        beta = 1/self.n
        
        population = self.init_population()
        
        for generation in range(self.max_gen):
            if time() - t0 > self.time_limit: # cutoff time
                print('Time limit exceeds, stops at generation', generation)
                break
            
            
            objectives = [max(self.objective(pop)) for pop in population]
            fitness = self.exponential(objectives, alpha=alpha, beta=beta)
            population = population[np.argsort(fitness)][::-1][:self.pop_size]
            fitness = np.sort(fitness)[::-1][:self.pop_size]
            
            gen_best.append(max(self.objective(population[0])))
            gen_worst.append(max(self.objective(population[-1])))
            
            pool = np.arange(self.pop_size)
        
        
        for i in range(0, self.parent_num, 2):
            parent_fitness = fitness[pool]
            i1, i2 = np.random.choice(pool, size=2, replace=False, p=parent_fitness/sum(parent_fitness))
            points = sorted(random.sample(range(self.n), self.cross_points))
            
            # k-point Crossover
            # o1, o2 = k_point_crossover(population[i1], population[i2], points)
            
            # cross_length = np.random.randint(n//3, n//2)  
            # start_gene = np.random.randint(n//2)
            # end_gene = start_gene + cross_length

            if self.cross_type:
                children = self.k_point_crossover(population[i1], population[i2], points=points)
            else:
                children = self.uniform_crossover(population[i1], population[i2])
            # Mutation
            
            for child in children:
                mutated = self.mutation(child)
                population = np.vstack((population, mutated))
                
            
            
            # population = np.vstack((population, o1, o2))
            pool = np.setdiff1d(pool, [i1, i2])
        
        
            
            if self.hybrid:
                for k in range(len(population)):
                    population[k] = self.contorlled_mutation(population[k])
            
        
        objectives = [max(self.objective(pop)) for pop in population]
        fitness = self.exponential(objectives, alpha=alpha, beta=beta)
        population = population[np.argsort(fitness)][::-1][:self.pop_size]

        T = [[] for _ in range(self.m)]
        for j in range(self.n):
            T[population[0][j]].append(j)
        
        C_best = self.objective(population[0])
        C_worst = self.objective(population[-1])
        
        if worst_ms:

            return T, C_best, C_worst, time()-t0
        else:
            
            return T, C_best, time()-t0



