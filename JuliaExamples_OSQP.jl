
#------------------------------------------------------------------------------
#=
#Demo, min 0.5x'P*x + q'x
#st. l<= Ax <= u

using OSQP
using SparseArrays

# Define problem data
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A = sparse([1. 1.; 1. 0.; 0. 1.])
l = [1.; 0.; 0.]
u = [1.; 0.7; 0.7]

# Create OSQP object
prob = OSQP.Model()

# Setup workspace and change alpha parameter
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u, alpha=1)

# Solve problem
results = OSQP.solve!(prob)
=#
#------------------------------------------------------------------------------
#=
# Demo

using OSQP
using SparseArrays

# Define problem data
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A = sparse([1. 1.; 1. 0.; 0. 1.])
l = [1.; 0.; 0.]
u = [1.; 0.7; 0.7]

# Create OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Solve problem
results = OSQP.solve!(prob)

# Update problem
q_new = [2.; 3.]
l_new = [2.; -1.; -1.]
u_new = [2.; 2.5; 2.5]
OSQP.update!(prob, q=q_new, l=l_new, u=u_new)

# Solve updated problem
results = OSQP.solve!(prob)
=#
#------------------------------------------------------------------------------
#=
# Demo

using OSQP
using SparseArrays, LinearAlgebra

# Define problem data
P = sparse([4. 1.; 1. 2.])
q = [1.; 1.]
A = sparse([1. 1.; 1. 0.; 0. 1.])
l = [1.; 0.; 0.]
u = [1.; 0.7; 0.7]

# Create OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Solve problem
results = OSQP.solve!(prob)

# Update problem
# NB: Update only upper triangular part of P
P_new = sparse([5. 1.5; 1.5 1.])
A_new = sparse([1.2 1.1; 1.5 0.; 0. 0.8])
OSQP.update!(prob, Px=triu(P_new).nzval, Ax=A_new.nzval)

# Solve updated problem
results = OSQP.solve!(prob)
=#
#------------------------------------------------------------------------------

#Huber fitting

using OSQP
using SparseArrays, LinearAlgebra, Random

# Generate problem data
Random.seed!(1)
n = 10
m = 100
Ad = sprandn(m, n, 0.5)
x_true = randn(n) / sqrt(n)
ind95 = rand(m) .> 0.95
b = Ad*x_true + 10*rand(m).*ind95 + 0.5*randn(m).*(1 .- ind95)

# OSQP data
Im  = sparse(I,m,m)
Om  = spzeros(m, m)
Omn = spzeros(m, n)
P = blockdiag(spzeros(n, n), 2*Im, spzeros(2*m, 2*m))
q = [zeros(m + n); 2*ones(2*m)]
A = [Ad  -Im -Im Im;
     Omn  Om  Im Om;
     Omn  Om  Om Im]
l = [b; zeros(2*m)]
u = [b; fill(Inf,2*m)]

# Create an OSQP object
prob = OSQP.Model()

#Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

#Solve problem
res = OSQP.solve!(prob)
#------------------------------------------------------------------------------

#LASSO

using OSQP
using SparseArrays, LinearAlgebra, Random

Random.seed!(1)
n = 10
m = 1000
Ad = sprandn(m, n, 0.5)
x_true = (randn(n) .> 0.8) .* randn(n) / sqrt(n)
#x_true = 1:n
b = Ad * x_true + 0.5 * randn(m)
gammas = range(1, stop=10, length=11)
# OSQP data
P = blockdiag(spzeros(n,n), sparse(I,m,m), spzeros(n, n))
q = zeros(2*n+m)
A = [Ad sparse(I,m,m)  spzeros(m,n);
    sparse(I,n,n) spzeros(n, m) -sparse(I,n,n);
    sparse(I,n,n) spzeros(n, m)  sparse(I,n,n)]
l = [b; fill(-Inf,n); zeros(n)]
u = [b; zeros(n); fill(Inf,n)]

# Create an OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Solve problem for different values of gamma parameter
for γ in gammas
    global res
    local q_new
    # Update linear cost
    q_new = [zeros(n+m); γ*ones(n)]
    OSQP.update!(prob, q=q_new)
    # Solve
    res = OSQP.solve!(prob)
end
#------------------------------------------------------------------------------

#LS problem

#min  0.5y'y
#st.
#y = Ax - b
# 0<=x<=1

using OSQP
using SparseArrays, LinearAlgebra, Random

Random.seed!(1)
m = 30
n = 20

Ad = sprandn(m, n, 0.7)
b  = randn(m)

# OSQP data
P = blockdiag(spzeros(n,n), sparse(I,m,m))
q = zeros(n+m)
A = [Ad -sparse(I,m,m);
     sparse(I,n,n) spzeros(n,m)]
l = [b; zeros(n)]
u = [b; ones(n)]

# Create an OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Solve problem
res = OSQP.solve!(prob)
#------------------------------------------------------------------------------

# Model predictive control

using OSQP
using SparseArrays, LinearAlgebra, Random

# Discrete time model of a quadcopter
Ad = [1       0       0   0   0   0   0.1     0       0    0       0       0;
      0       1       0   0   0   0   0       0.1     0    0       0       0;
      0       0       1   0   0   0   0       0       0.1  0       0       0;
      0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
      0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
      0       0       0   0   0   1   0       0       0    0       0       0.0992;
      0       0       0   0   0   0   1       0       0    0       0       0;
      0       0       0   0   0   0   0       1       0    0       0       0;
      0       0       0   0   0   0   0       0       1    0       0       0;
      0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
      0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
      0       0       0   0   0   0   0       0       0    0       0       0.9846]
Bd = [0      -0.0726  0       0.0726;
     -0.0726  0       0.0726  0;
     -0.0152  0.0152 -0.0152  0.0152;
      0      -0.0006 -0.0000  0.0006;
      0.0006  0      -0.0006  0;
      0.0106  0.0106  0.0106  0.0106;
      0      -1.4512  0       1.4512;
     -1.4512  0       1.4512  0;
     -0.3049  0.3049 -0.3049  0.3049;
      0      -0.0236  0       0.0236;
      0.0236  0      -0.0236  0;
      0.2107  0.2107  0.2107  0.2107]
(nx, nu) = size(Bd)


# Constraints
u0 = 10.5916;
umin = [9.6; 9.6; 9.6; 9.6] .- u0
umax = [13; 13; 13; 13] .- u0
xmin = [-pi/6; -pi/6; -Inf; -Inf; -Inf; -1; fill(-Inf,6)]
xmax = [ pi/6;  pi/6;  Inf;  Inf;  Inf; Inf; fill(Inf,6)]

# Objective function
Q  = diagm([0,0,10,10,10,10,0,0,0,5,5,5])
QN = copy(Q)
R  = 0.1*Matrix(I(4))

# Initial and reference states
x0 = zeros(12)
xr = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0]

# Prediction horizon
N = 10

#Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
# - quadratic objective
P = blockdiag( kron(sparse(I,N,N), Q), sparse(QN), kron(sparse(I,N,N), R) )
# - linear objective
q = [repeat(-Q*xr, N); -QN*xr; zeros(N*nu)]
# - linear dynamics
Ax = kron(sparse(I,N+1,N+1),-sparse(I,nx,nx)) + kron(sparse(diagm(-1 => ones(N))), Ad)
Bu = kron([spzeros(1, N); sparse(I,N,N)], Bd)
Aeq = [Ax Bu]
leq = [-x0; zeros(N*nx)]
ueq = copy(leq);
# - input and state constraints
Aineq = sparse(I,(N+1)*nx + N*nu,(N+1)*nx + N*nu)
lineq = [repeat(xmin, N+1); repeat(umin, N)]
uineq = [repeat(xmax, N+1); repeat(umax, N)]
# - OSQP constraints
A = [Aeq; Aineq]
l = [leq; lineq]
u = [ueq; uineq]

# Create an OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Simulate in closed loop
nsim = 15
for i = 1:nsim
    global res, x0, l ,u
    local ctrl
    # Solve
    res = OSQP.solve!(prob)
    # Check solver status
    if res.info.status != :Solved
        error("OSQP did not solve the problem!")
    end

    # Apply first control input to the plant
    ctrl = res.x[(N+1)*nx+1:(N+1)*nx+nu]
    x0 = Ad*x0 + Bd*ctrl

    # Update initial state
    l[1:nx] = -x0
    u[1:nx] = -x0
    OSQP.update!(prob,l=l, u=u)
end
#------------------------------------------------------------------------------

# Portfolio optimization

using OSQP
using SparseArrays, LinearAlgebra, Random

# Generate problem data
Random.seed!(1)
n = 100
k = 10
F = sprandn(n, k, 0.7)
D = sparse( diagm(sqrt(k)*rand(n)) )
mu = randn(n)
gamma = 1

# OSQP data
P = blockdiag(D, sparse(I,k,k))
q = [-mu/(2*gamma); zeros(k)]

A = [F' -sparse(I,k,k);
     ones(1, n) zeros(1, k);
     sparse(I,n,n) spzeros(n, k)]
l = [zeros(k); 1; zeros(n)]
u = [zeros(k); 1; ones(n)]

# Create an OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Solve problem
res = OSQP.solve!(prob)
#------------------------------------------------------------------------------

# Support vector machines

using OSQP
using SparseArrays, LinearAlgebra, Random

# Generate problem data
Random.seed!(1)
n = 10
m = 1000
N = cld(m,2)
gamma = 1
A_upp = sprandn(N, n, 0.5)
A_low = sprandn(N, n, 0.5)
Ad = [A_upp / sqrt(n) + (A_upp .!= 0) / n;
      A_low / sqrt(n) - (A_low .!= 0) / n]
b = [ones(N); -ones(N)]

# OSQP data
P = blockdiag(sparse(I,n,n), spzeros(m, m))
q = [zeros(n); gamma*ones(m)]
A = [diagm(b)*Ad   -sparse(I,m,m);
     spzeros(m, n)  sparse(I,m,m)]
l = [-Inf*ones(m); zeros(m)]
u = [-ones(m); Inf*ones(m)]

# Create an OSQP object
prob = OSQP.Model()

# Setup workspace
OSQP.setup!(prob; P=P, q=q, A=A, l=l, u=u)

# Solve problem
res = OSQP.solve!(prob)
#------------------------------------------------------------------------------
