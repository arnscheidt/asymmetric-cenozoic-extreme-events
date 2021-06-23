# numerical stochastic climate-carbon cycle model 
# from Arnscheidt and Rothman - Asymmetry of extreme Cenozoic climate-carbon cycle events (2021)

using DifferentialEquations
using Plots
using BenchmarkTools
using DelimitedFiles

# unit convention: SI, except for mass (Pg) and time (years)

# PARAMETERS

# 1. carbonate chemistry approximation parameters
const γ = 6.5
const Iᵦ= 58000.0

# 2. isotopic parameters 

const vpdb = 0.01123720 	# ratio of 13C/12C in VPDB standard
const δᵪ = 1.0 			# isotopic composition of carbonate carbon
const Rχ = (δᵪ/1000 + 1)*vpdb 	# ratio of 13C/12C in carbonate carbon
const δₒ = -25
const Rₒ = (δₒ/1000 + 1)*vpdb

# 4. other parameters
const Myr = 1.0*10^6
const kyr = 1000.0
const sinyr = 365.25*24*3600.0	# seconds in a year
const P₀ = 400.0*10^(-6)  	# pCO₂ in reference state
const I₀ = 38000.0		# reference surficial carbon stock
const τ = 100kyr		# timescale of weathering feedback
const λ = 5.0 			# Earth System Sensitivity (K)
const a = 2.2*sinyr		# slope of OLR with T (J/yr/m²/K)
const C = 2.0*10^(8)		# heat capacity of the Earth (J/m²/K)

end 

# FUNCTION SPECIFICATION

function model!(du,u,par,t) # the model equations

	# define explicitly for ease of understanding
	ΔT = u[1]
	I = u[2]
	I₁₃ = u[3]

	if I<0
		I = 0.0
	end

	P = ((I^γ)/(I^γ+Iᵦ^γ))*10^(-6)*7000.0

	du[1] = (1/C)*(-a*(ΔT-par[5](t))+a*λ*log(P/P₀)/log(2))  
	du[2] = - (I-I₀)/τ
	du[3] = - (I₁₃-I₀*Rχ)/τ
	du[4] = 0

end

function noise!(du,u,par,t)

	du[1,1] = par[1]*(u[1]+par[3])
	du[2,1] = par[2]*(u[1]+par[3])
	du[3,1] = par[2]*(u[1]+par[3])*Rₒ
	du[4,1] = par[2]*(u[1]+par[3])

	du[1,2] = par[4]
	du[2,2] = 0 
	du[3,2] = 0 
	du[4,2] = 0
end

# ENSEMBLE RUN - periodic forcing
u0 = [0.0,I₀,Rχ*I₀,0.0]
tspan = (0.0,10Myr)
tstep = 10000 
t = range(tspan[1],tspan[2],step=tstep)

F = t->0.6(sin(2*π*t/(400000))) 

νₜ= 0.2
νᵪ= 1.0
c = 1.0 
μ = 0.4

prob_st = SDEProblem(model!,noise!,u0,tspan,[νₜ,νᵪ,c,μ,t->F(t)],noise_rate_prototype=zeros(4,2))
ensemble_prob = EnsembleProblem(prob_st)
sol = solve(ensemble_prob,dt=1,saveat=tstep,EM(),trajectories=10)

# convert sol[3,:,:] to be time series of d13c
sol[3,:,:] = ((sol[3,:,:]./sol[2,:,:])/vpdb.-1)*1000

# write outputs to file
#writedlm("ens_400_T.csv",sol[1,:,:],',')
#writedlm("ens_400_dc.csv",sol[3,:,:],',')



