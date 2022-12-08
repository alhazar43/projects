<center><h1> MAST 90014 - Group 5 Project </h1></center>
<center>Wenrui Yuan, Raelene Huang, Carlos Davalos, Chenyang Zhuang, Mingjun Yin </center>
<center><h2>Home service problem with VRPTW formulation</h2></center>

---

### 1. Introduction
We present a home  service allocation problem with a particular interest in the  routing allocation of service providers to find the best optimal routes that yield least hiring costs. In particular, this project is presented in two parts:

> - the *basic model*, defined in `task1.jl`
> - the *improved model*, made available in `model.jl`
> 
Additionally, two set of instance [30C-20P.xlsx](30C-20P.xlsx) and `16C-10P.xlsx` are **only** for `model.jl`. If you wish to test on a smaller instance, please change the loading filename as stated below accordingly:
```Julia
filename = "30C-20P.xlsx"
```

### 2. Dependencies
This project is built with the following packages:

>    - [JuMP](https://jump.dev/JuMP.jl/v0.21.1/installation/)
>    - [Gurobi](https://github.com/jump-dev/Gurobi.jl)
>    - [XLSX](https://felipenoris.github.io/XLSX.jl/stable/)

Please make sure that you have follow the guide and installed these packages before running any code files.

### 3. Model result & project report
Larger instances have not been tested fro `task.jl`, but please be warned that the runtime for a feasible solution may be significant, which is intended and clearly stated in our report, `report.pdf`. Additionally, you may refer to the `results` folder for the three main results derived from both models.
