from pathlib import Path
from tqdm import tqdm

import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation

import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.spatial import distance




class SandPile():
    BOUND = 1 # simulation boundary size
    """SandPile class
    """
    def __init__(self, width=3, height=3, threshold=4, stochastic=False, abelian=True):
        """Initialize a sandpile with the specified attributes.
            Attributes:
                width (int): width of the sandpile, default to 3 (minimum simulation grid)
                height (int): height/length of the sandpile, default to 3
                threshold (int): threshold for instability condition
            
            Types:
                abelian (bool): select Abelian model if True; non-Abelian model otherwise
                stochastic (bool): select stochastic sandpile model if True
        """
        self.width = width
        self.height = height
        self.threshold = threshold
        
        self.stochastic = stochastic
        self.abelian = abelian
        self.grid = np.zeros((width+2*self.BOUND, height+2*self.BOUND), dtype=int)

        self.loss_history = []
        self.area_history = []
        self.size_history = []
        self.time_history = []
        self.length_history = []
        self.distance_history = []
        self.avg_height_hist = []
        self.mass_history = []
        self.ims = []

    def drive(self, drops=1, site=None, gaussian=None, animate_every=0, save_anim=False):
        """Drive the system to steady state with specificed arguments
            Args:
                drops (int): grains to drop, default to 1

                site (list): specificed site to drop all grains. e.g. site=[25,25]
                
                gaussian (2D list): normally distribute sands near a specified site 
                                    with a specificed sigma e.g. gaussian=[[25,25],5]
                                    
                animate_every (int): animation speed, k = animte every k drops; 
                                     0 for not display animation

                save_anim (bool): save animation       
        """
        out_dir = Path.cwd() / "output"
        fig = plt.figure()
        if not out_dir.is_dir():
            out_dir.mkdir()
        for k in tqdm(range(drops), colour='green'):
            # Passing arguments
            self.drop(site=site, gaussian=gaussian)
            self.avalanche(animated=(animate_every and k % animate_every==0))

        if animate_every:     
            ani = animation.ArtistAnimation(fig, self.ims, interval=100, repeat=False, blit=True)
            if save_anim:
                ani.save(out_dir/'sandpile.mp4', fps=30)
            plt.show()

    
    def drop(self, site=None, gaussian=None):
        """Drop a grain of sand on specified site; drop randomly if unspecified
            Args:
                site (list): same as in drive method
                gaussian (2D list): same as in drive method
        """
        # np.random.seed(42) # Fix seed for reproduction
        if site:
            self.grid[site[0]+self.BOUND, site[1]+self.BOUND] += 1
        elif gaussian:
            # Set upper and lower bounds
            i = min(max(int(round(np.random.normal(gaussian[0][1], gaussian[1]))), 1), self.width)
            j = min(max(int(round(np.random.normal(gaussian[0][1], gaussian[1]))), 1), self.height)
            self.grid[i,j] += 1
        else:
            i = np.random.randint(self.BOUND, self.width+self.BOUND)
            j = np.random.randint(self.BOUND, self.height+self.BOUND)
            self.grid[i,j] += 1

    @property
    def avalanche_loss(self):
        return self.loss_history

    @property
    def avalanche_area(self):
        return self.area_history

    @property
    def avalanche_size(self):
        return self.size_history

    @property
    def avalanche_length(self):
        return self.length_history


    def avalanche(self, animated=False):
        """Avalanche loop for different models
            Args:
                animated (bool): argumented passed by drive funtion, 
                                 display animation if True
        """
        # Intialize metrics
        lifetime = 0
        size = 0
        area = np.zeros_like(self.grid, dtype=bool)
        loss = 0
        l = []
        
        # Abelian model
        if self.abelian:
            # Search for unstable sites
            z0 = np.argwhere(self.grid >= self.threshold)
            
            while np.max(self.grid) >= self.threshold:
                i,j = np.where(self.grid >= self.threshold)

                # Statistics to collect
                size += i.shape[0] # Avalanche Size
                area[i,j] = True # Avalanche Area
                lifetime += 1 # Avalanche duration
                z1 = np.dstack((i,j))[0]
                l.append(distance.cdist(z1,z0,'cityblock').max()) # Avalanche Length
                
                # Toppling
                self.grid[i,j] -= self.threshold

                if self.stochastic:
                    neighbors = np.random.choice(np.array((-1, 0, 1)), size=(self.threshold,2))
                else:
                    neighbors = [[-1,0],[1,0],[0,-1],[0,1]]
                    
                for neighbor in neighbors:
                    self.grid[i+neighbor[0],j+neighbor[1]] += 1

                # Avalanche loss
                loss += self.grid[[0,-1]].sum() + self.grid[1:-1,[0,-1]].sum()

                # Reset ovespill on padding sites
                self.grid[0] = self.grid[-1] = 0
                self.grid[:, 0] = self.grid[:, -1] = 0
        # non-Abelian model
        else:
            # Compute Local gradients of all sites
            grad_s = signal.convolve2d(self.grid, np.array([[0,-1],[0,1]]), mode='valid')
            grad_e = signal.convolve2d(self.grid, np.array([[0,0],[-1,1]]), mode='valid')
            grad_w = signal.convolve2d(self.grid, np.array([[0,0],[1,-1]]), mode='valid')
            grad_n = signal.convolve2d(self.grid, np.array([[0,1],[0,-1]]), mode='valid')
            # grad_se = signal.convolve2d(self.grid, np.array([[-1,0],[0,1]]), mode='valid')

            # Search for unstable sites
            z0_s = np.argwhere(grad_s>3)
            z0_e = np.argwhere(grad_e>3)
            z0_w = np.argwhere(grad_w>3)
            z0_n = np.argwhere(grad_n>3)
            # z0_se = np.argwhere(grad_se>3)

            # Shift index to offset padding
            z0_w[:,1] += 1
            z0_n[:,0] += 1

            # Randomly selecting a candidate (to avalanche)
            z0 = [z0_n, z0_e, z0_s, z0_w]
            c_k = []
            for k, e in enumerate(z0):
                if e.any():
                    c_k.append(k)
            if c_k:
                c = np.random.choice(np.array(c_k))
                z0 = z0[c]

            while np.max(grad_w)>3 or np.max(grad_s)>3 or np.max(grad_e)>3 or np.max(grad_n)>3:
                sites_s = np.array(np.where(grad_s>3))
                sites_e = np.array(np.where(grad_e>3))
                sites_w = np.array(np.where(grad_w>3))
                sites_n = np.array(np.where(grad_n>3))
                sites_w[1] += 1
                sites_n[0] += 1
                # sites_se = np.array(np.where(grad_se>4))
                

                directed = [sites_n, sites_e, sites_s, sites_w]
                i, j = directed[c]       
                z1 = np.dstack((i,j))[0]
                if not z1.any():
                    z1 = z0
                
                # Statistics to collect
                area[i,j] = True
                lifetime += 1
                size += i.shape[0]
                l.append(distance.cdist(z1,z0,'cityblock').max())

                # Toople
                self.grid[i,j] -= 4
                self.grid[i+1,j] += 1
                self.grid[i-1,j] += 1
                self.grid[i,j+1] += 1
                self.grid[i,j-1] += 1
                # self.grid[i+1,j+1] += 1

                # Randomly choosing (next) candidates
                c_k = []
                for k, e in enumerate(directed):
                    if e.any():
                        c_k.append(k)
                c = np.random.choice(np.array(c_k))

                # Collect energy loss
                loss += self.grid[[0,-1]].sum() + self.grid[1:-1,[0,-1]].sum()

                # Reset ovespill on padding sites
                self.grid[0] = self.grid[-1] = 0
                self.grid[:, 0] = self.grid[:, -1] = 0

                # Update for the next search
                grad_s = signal.convolve2d(self.grid, np.array([[0,-1],[0,1]]), mode='valid')
                grad_e = signal.convolve2d(self.grid, np.array([[0,0],[-1,1]]), mode='valid')
                grad_w = signal.convolve2d(self.grid, np.array([[0,0],[1,-1]]), mode='valid')
                grad_n = signal.convolve2d(self.grid, np.array([[0,1],[0,-1]]), mode='valid')
                # grad_se = signal.convolve2d(self.grid, np.array([[-1,0],[0,1]]), mode='valid')
        
        # Average height of the pile
        self.avg_height_hist.append(self.grid.sum()/(self.width*self.height))

        # Record non-zero stastics
        if lifetime:
            self.size_history.append(size)
            self.length_history.append(max(l))
            self.area_history.append(np.count_nonzero(self.remove_bound(area)))
            self.time_history.append(lifetime)
            self.loss_history.append(loss)
            self.mass_history.append(self.grid.sum())

        if animated:
            im = plt.imshow(self.remove_bound(self.grid), 
                            interpolation='none', 
                            cmap='copper', 
                            animated=True
                           )
            plt.axis('off')
            self.ims.append([im])
    
    @classmethod
    # Remove simulation boundary
    def remove_bound(cls, grid):
        return grid[cls.BOUND:-cls.BOUND,cls.BOUND:-cls.BOUND]


    def plot_shape(self, save_plot=True):
        """Plot limiting shape of the sandpile using heatmap
            Args:
                save_plot (bool): save sandpile plot
        """
        out_dir = Path.cwd() / "output"
        if not out_dir.is_dir():
            out_dir.mkdir()
        fig, ax = plt.subplots(figsize=(10,10))
        c = ax.imshow(self.remove_bound(self.grid), cmap='copper')
        fig.colorbar(c, ticks=range(self.grid.max()+1))
        plt.show()
        
        if save_plot:
            if self.abelian:
                if self.stochastic:
                    filename = 'randT_sandpile.pdf'
                else:
                    filename = 'abelian_sandpile.pdf'
            else:
                filename = 'non_abelian_sandpile.pdf'
            fig.savefig(out_dir/filename, bbox_inches='tight')
    

    def plot_3D(self, save_plot=False):
        """Plot limiting shape of the sandpile in 3D based on height
            Args:
                save_plot (bool): save sandpile 3D plot
        """
        fig = plt.figure(figsize=plt.figaspect(2.))
        ax = fig.add_subplot(2, 1, 1, projection='3d')
        X = np.linspace(0, self.width)
        Y = np.linspace(0, self.height)
        X, Y = np.meshgrid(X,Y)
        Z = self.remove_bound(self.grid)
        ax.plot_surface(X,Y,Z, rstride=1, cstride=1,
                cmap='copper', edgecolor='none')
        ax = fig.add_subplot(2, 1, 2)
        ax.imshow(self.remove_bound(self.grid), cmap='copper')        
        if save_plot:
            out_dir = Path.cwd()/'output'
            if not out_dir.is_dir():
                out_dir.mkdir()
            if self.abelian:
                if self.stochastic:
                    filename = 'randT_sandpile_3D.pdf'
                else:
                    filename = 'abelian_sandpile_3D.pdf'
            else:
                filename = 'non_abelian_sandpile_3D.pdf'
            fig.savefig(out_dir/filename, bbox_inches='tight')
        plt.show()

    # Plot average height
    def plot_average_height(self):
        fig, ax = plt.subplots(figsize=(5,5))
        x = range(1, len(self.avg_height_hist)+1)
        x_b = stats.mode(self.avg_height_hist)[0]
        ax.plot(x, self.avg_height_hist)
        ax.axhline(x_b, color='r', label=r'$h=%.2f$' % x_b)
        ax.set_xlabel(xlabel=r'$Drops$', fontsize=18)
        ax.set_ylabel(ylabel=r'$Average\,\,height$', fontsize=18)
        ax.legend(loc='best', fontsize=14, fancybox=True)
        plt.show()


    def plot_stats(self, save_plot=True, save_csv=False):
        """Plot frequency distributions and make power-law fit for each 
            Args:
                save_plot (bool): save distribution plots
                save_csv (bool): save raw stats (for debug)
        """
        out_dir = Path.cwd()/'output'
        if not out_dir.is_dir():
            out_dir.mkdir()

        # Statistics to plot
        stats = [self.size_history, self.area_history, self.loss_history, self.length_history]
        labels = ['Size', 'Area', 'Loss', 'Length']

        fig, ax = plt.subplots(1,len(stats), constrained_layout=True, figsize=(5*len(stats),5))
        for i, stat in enumerate(stats):
            data, counts = np.unique(stat, return_counts=True)
            pdf = counts/counts.astype(np.float64).sum()

            # Linear fit near the tail
            try:
                idx = np.nanargmin(np.abs(pdf-np.percentile(pdf, 90-20*(i//2))))
                coeff = np.polyfit(np.log10(data[1:idx]), np.log10(pdf[1:idx]),1)
            except TypeError: # Catch poor fit
                idx = -1
                coeff = np.polyfit(np.log10(data[1:idx]), np.log10(pdf[1:idx]),1)
                
            ax[i].loglog(data[data>0], pdf[data>0],
                                 '.',
                                 color='gray',
                                 alpha=0.7,
                                 label=r'Data'
                                )
            ax[i].loglog(data[data>0],
                                 10**(np.polyval(coeff, np.log10(data[data>0]))), 
                                 color='r', 
                                 linewidth=2,
                                 label=r'$a=%.2f, k=%.2f$'%(-coeff[1],coeff[0])
                                )            
            ax[i].set_xlabel(xlabel=r'$Avalanche\,\,%s$' % labels[i], fontsize=18)
            ax[i].set_xlim([data[data>0].min(), data[data>0].max()])
            ax[i].legend(loc='best', fontsize=14, fancybox=True)
            ax[i].set_ylim([10e-6,10**(np.polyval(coeff, np.log10(data[data>0]))).max()])

            if save_csv:
                pd.DataFrame(data={'data':data,'pdf':pdf}).to_csv(out_dir/(labels[i]+'.csv'), index=False)

        if save_plot:
            if self.abelian:
                if self.stochastic:
                    filename = 'randT_sandpile_stats.pdf'
                else:
                    filename = 'abelian_stats.pdf'
            else:
                filename = 'non_abelian_stats.pdf'
            
            fig.savefig(out_dir/filename, bbox_inches='tight')
            
        plt.show()