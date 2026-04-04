module SingleDMRGLMG

using ITensors
using ITensorMPS
using CUDA

export run_dmrg_lmg_single

#=
Created by Maggie Bao 8/20/2025
    
run_dmrg_lmg_single(N, ε, V, W;
                        nsweeps=5,
                        maxdim=[10,20,100,100,200],
                        cutoff=1e-10,
                        use_gpu=false)

Solves the spin-1/2 LMG model with DMRG on a set of parameters and returns `(energy, elapsed_seconds)`.
Timing is DMRG-only, so the state/hamiltonian construction runtime is excluded.
=#
function run_dmrg_lmg_single(
    N::Int, ε::Float64, V::Float64, W::Float64;
    nsweeps::Int=5,
    maxdim::Vector{Int}=[10,20,100,100,200],
    cutoff::Float64=1e-10,
    use_gpu::Bool=false,
)
    @assert N ≥ 2 "N must be ≥ 2"

    # Build sites and hamiltonian

    sites = siteinds("S=1/2", N) # LMG model standard: formulated as N spin-1/2 particles
    os = OpSum()
    for i in 1:N
        os += (ε/2, "Z", i)
    end
    @inbounds for i in 1:N-1, j in i+1:N
        os += ( V/2, "X", i, "X", j)
        os -= ( V/2, "Y", i, "Y", j)
        os += ( W/2, "X", i, "X", j)
        os += ( W/2, "Y", i, "Y", j)
    end
    H = MPO(os, sites)

    # Choose random intial state with small link dimension, growth controlled by maxdim
    psi0 = randomMPS(sites; linkdims=min(4, maxdim[1]))

    #GPU compatibility: convert to cuArrays
    if use_gpu
        CUDA.allowscalar(false)
        H    = cu(H)
        psi0 = cu(psi0)
    end

    t0 = time_ns()
    # outputlevel=0 silences logs (supported by ITensors DMRG)
    energy, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    elapsed_s = (time_ns() - t0) / 1e9
    return energy, elapsed_s
end

end #module
