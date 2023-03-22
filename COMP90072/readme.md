# Generalized Sandpile Models

This project is created as a part of submission for the subject COMP90072: The Art of Scientific Compuation. 
> Before running any files in this package, please refer to [package.txt](package.txt) to install or update all required packages.

## 1. Model structure and use cases

Use [main.py](main.py) to perform simulation. In case there are any conflicting packages, please consider run it in a virtual environment. Main body of this model is in [sandpile.py](sandpile.py), which is composed of 2 parts, simulation and plotting; if a error is flagged when simulating, please make sure that a correct **model type** is specified. For plotting errors, consider either **increase** the grain drops or **decrease** the grid size.

### 1.1 Use cases
To run the simulation, initialize a new ```Sandpile``` type with specified grid size ```width=n```, ```height=m``` and model type, which can be taken from one of the following:
- abelian: Run simulation with the Abelian sandpile model if ```abelian=True```; default to run this model if unspecified.
- stochastic: Run simulation with the stochastic sandpile model if ```stochastic=True```. Please do not run when ```abelian=False```; these two types must be both ```True``` for simulation to run.
- non-abelian: Run simulation with the non-Abelian sandpile if ```abelian=False```, incompatible with ```stochastic=True```

> Sample usage:
> ```Python
> # intialize the grid with Abelian configuration
> pile = SandPile(50,50, abelian=True) 
> ```
> ```Python
> # intialize the grid with stochastic configuration
> pile = SandPile(50,50, abelian=True, stochstic=True) # or
> pile = SandPile(50,50, stochstic=True)
> ```
> ```Python
> # intialize the grid with non-abelian configuration
> pile = SandPile(50,50, abelian=False)
> ```

### 1.2 Main function
#### 1.2.1 ```drive```
Set the system to drive itself to criticality by specifing
- ```drops``` (integer), it is recommended that this value is much greater than the area of the grid to get optimal results.
- ```site``` (list) specificed site to drop all grains. e.g.
- ```gaussian``` (2D list) normally distribute sands near a specified site, and a $\sigma^2$ value to control the spread.
- ```animate_every``` (int) animation speed, $k$ = animte every $k$ drops of sand; 0 for not display animation
- ```save_anim``` (bool) save animation if ```True```; must be ```False``` if ```animate_every=0```

> Sample usage:
> ```Python
> pile = SandPile(50,50, abelian=True)
> # drive the system with 50,000 grains of sand dropped at [25,25]
> # and save snapshots every 10 drops to produce a animation
> pile.drive(drops=50000, site=[25,25], animate_every=10, save_anim=True)
> ```
> ```Python
> pile = SandPile(50,50, stochastic=True)
> # drive the system with 50,000 gaussian drops near [25,25]
> pile.drive(drops=50000, gaussian=[[25,25],5], animate_every=0, save_anim=False)
> ```

#### 1.2.2 ```plot``` functions
There are a few plotting method of this model; they are ```plot_3D```, ```plot_shape``` and ```plot_stats```. All of them share the same argument:
- ```save_plot``` (bool) save corresponding plot.

Also, please do ignore ```save_csv``` in the ```plot_stats``` method as it is for debugging only. If an error is raised, consider increase ```drops``` in the ```drive``` method first.\
Block out ```plot_stats``` for intermediate results for ```plot_3D``` and ```plot_shape```.\
There's also another plotting method ```plot_average_height```, which is poorly implemented (may cause potential bugs) and should be used for verification purpose with Abelian models only.

#### 1.2.3 classmethod and class properties
There is a convinient function to remove the simulation boundary called ```remove_bound``` and is not recommended to directly interact with the [main.py](main.py) file.\
A few properties can be returned by calling them directly in [main.py](main.py):
```Python
pile.avalanche_loss # return avlanche loss data
pile.avalanche_area # return avlanche area data
pile.avalanche_length # return avlanche length data
pile.avalanche_size # return avlanche size data
```

## 2. Other remarks
If all required packages are set up as instructed, then the only exception will raise is when plotting the non-abelian model in a centered drop setting. It is a observed property as stated in the final report. Although ```TypeError``` is catched, when it comes to collecting satistics, it is always recommended to run the model with more grains and lower grid size.\
Here are a few suggested minimal setup for plotting stats:
> Abelian and stochastic:
> ```Python
> pile = SandPile(50,50, abelian=True)
> pile.drive(drops>10000) # for display purpose only, please assign an integer value to it
> ```
> Non-Abelian:
> ```Python
> pile = SandPile(50,50, abelian=False)
> pile.drive(drops>50000) # for display purpose only, please assign an integer value to it
> ```
