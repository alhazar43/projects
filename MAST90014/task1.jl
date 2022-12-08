using JuMP, Gurobi


#Data Example 1

#N0: set of nodes
N0 = [1 2 3 4 5 6 7]
#N: set of clients
N = [2 3 4 5 6 7]
#Service start per client
s = [4 1 4 6 5 3]
#Service duration per client (Add 0 to "provide service to headquarters")
d = [0 2 1 1 1 1 2]
#M: set of service providers
M = [1 2 3 4 5]
#Hiring cost per provider
f = [15 10 8 12 15]
#Time provider starts day
w = [0 3 4 0 2]
#Time from client i to j
t = [0 3 1 1 1 1 2; 2 0 2 3 4 3 2; 1 2 0 1 2 3 2 ;1 3 1 0 1 4 4 ; 1 4 3 1 0 1 2 ;1 3 3 2 1 0 1; 1 1 2 3 3 1 0]
#Hiring cost per provider
f = [15 10 8 12 15]


#N0: set of nodes
N0 = [1 2 3 4 5 6 7 8 9 10 11]
#N: set of clients
N = [2 3 4 5 6 7 8 9 10 11]
#Service start per client
s = [4 1 4 6 5 5 2 3 1 2]
#Service duration per client (Add 0 to "provide service to headquarters")
d = [0 2 1 1 1 1 2 3 4 1 1]
#M: set of service providers
M = [1 2 3 4 5 6 7 8 9]
#Hiring cost per provider
f = [15 10 8 12 15 7 6 9 18]
#Time provider starts day
w = [0 3 4 0 2 2 4 1 0]
#Time from client i to j
t = [0 3 1 1 1 1 2 2 1 3 1; 2 0 2 3 2 3 2 3 1 2 1; 1 2 0 1 2 3 2 1 2 2 2; 1 3 1 0 1 2 2 1 3 1 2;
    1 2 3 1 0 1 2 2 2 2 3; 1 3 3 2 1 0 1 2 1 2 2; 1 1 2 3 3 1 0 2 1 1 3; 1 1 1 1 2 3 2 0 3 1 1;
    2 2 1 3 3 4 2 2 0 2 3; 1 1 2 1 3 2 1 1 2 0 1; 1 3 2 3 1 1 1 1 3 1 0]
#Time day ends
l = 13
#Big M
BM = 99999

using JuMP, Gurobi

model = Model();
set_optimizer(model, Gurobi.Optimizer);

#Indicate whether provider k(K) visits i(N) before j(N)
@variable(model, x[N0, N0, M], Bin)

#time that the provider k(M) visits client i(N)
@variable(model, c[N0, M], Int)


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

# 8. Provider should return to headquarters before the end of the day
@constraint(model, cons8[i in N0, j in N, k in M], c[i,k] + d[i] + t[i,1] <= l)


# Provider finish work before the end of the day
@constraint(model, conss[i in N0, k in M], w[k] <= c[i, k] <= l)


@objective(model, Min, sum( x[1,j,k]*f[k] for j in N for k in M) )

optimize!(model)

for v in M, i in N0, j in N0
    if value(x[i,j,v]) == 1
        println("start $i, end $j, with provider $v at time: ", round(value(c[j,v]),digits = 0),
                                            ", distance: ", value(t[i,j]),
            " service time at client $j: ", d[j])
    end
end
println("")

