using JuMP, Gurobi, XLSX

# =================================================================================================
#                                              Data
# =================================================================================================
# Create relative path
filename = "30C-20P.xlsx" # Choose which instance of data should be loaded
filepath = joinpath(@__DIR__, filename)
# Read data from spreadsheets
data = XLSX.readxlsx(filepath);
General = data["General"];
Clients = data["Clients"];
Providers = data["Providers"];
Distances = data["Distances"];

# Define indicies
m = General["B1"] # number of providers
n = General["B2"]+1 # number of nodes (including depot)
N = 2:n # set of clients
N0 = 1:n # set of all nodes
M = 1:m # set of providers

# Clients data
s_start = Clients[string("A2:A", n)] # availability of each client
s_end = Clients[string("B2:B", n)] # service end time for each client
d = cat(0, Clients[string("C2:C", n)], dims=1) # service duration for each client
s_skills = Clients[string("D2:G", n)] # skills requested by each client

# Set of skills
H = 1:size(s_skills,2) # dimesion of skills 

# Providers data
w_start = Providers[string("B2:B", m+1)] # availability of each provider
w_end = Providers[string("C2:C", m+1)] # service end time for each provider
f = Providers[string("A2:A", m+1)] # hourly hiring cost of each provider
w_skills = Providers[string("D2:G", m+1)] # set of skills equipped by each provider

# Distance matrix marked by travel time
t = Distances[Distances.dimension]

#Big M for linearisation
BM = 99999

# =================================================================================================
#                                              Model
# =================================================================================================
# Definine variables and the model
model = Model();
set_optimizer(model, Gurobi.Optimizer);

# Binary variable, indicate whether provider k(M) moves from client i(N0) to client j(N0)
@variable(model, x[N0, N0, M], Bin)

# Integer variable, record time that the provider k(M) visits client i(N0) 
@variable(model, c[N0, M], Int)

# Objective funcion - minimizing hourly costs
@objective(model, Min, sum( x[i,j,k]*f[k]*d[j] for i in N0 for j in N for k in M) )

# ------------------------------------------Constraints--------------------------------------------
# 1. Each provider leaves headquarters (0)
@constraint(model, cons1[k in M], sum(x[1, j, k] for j in N) <= 1)

# 2. Each provider ends at headquarters (0)
@constraint(model, cons2[k in M], sum(x[i, 1, k] for i in N) <= 1)

# 3. Each provider leaves the client it visited
@constraint(model, cons3[h in N, k in M], 
            sum(x[i, h, k] for i in N0) - sum(x[h, j, k] for j in N0) == 0)

# 4. Each client is visited exactly once by one provider
@constraint(model, cons4[j in N], 
            sum( x[i, j, k] for i in N0 for k in M) == 1)

# 5. Complete service
@constraint(model, cons5[i in N0, j in N, k in M], 
            c[i,k] + d[i] + t[i,j] - BM*(1 - x[i, j, k]) <= c[j,k])

# 6. Start time for each provider
@constraint(model, cons6[k in M], w_start[k] <= c[1, k])

# 7. Client service time window
@constraint(model, cons7[j in N, k in M], s_start[j-1] <= c[j, k] <= s_end[j-1])

# 8. Provider should return to headquarters before the end of the day
@constraint(model, cons8[i in N0, j in N, k in M], c[i,k] + d[i] + t[i,1] <= w_end[k])        

# 9. Matching skill demands
@constraint(model, cons9[j in N], 
            sum(x[i, j, k]*w_skills[k, h]*s_skills[j-1, h] for i in N0 for k in M for h in H)==1)
# -------------------------------------------------------------------------------------------------

optimize!(model)

# =================================================================================================
#                                              Results
# =================================================================================================

for v in M, i in N0, j in N0
    if value(x[i,j,v]) == 1
        println("Start $(i-1), end $(j-1), with provider $v at time: ", 
                round(value(c[j,v]),digits = 0), ", distance: ", value(t[i,j]),
                " service time at client $(j-1): ", d[j], " hr(s).")    
    end
end

