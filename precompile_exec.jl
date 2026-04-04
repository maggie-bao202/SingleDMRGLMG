# precompile_exec.jl
#= 
Created by Maggie 9/22/2025
Goal: precompile SingleDMRGLMG with small load
=#

include("src/SingleDMRGLMG.jl")
using .SingleDMRGLMG
using CUDA

# CPU representative call
SingleDMRGLMG.run_dmrg_lmg_single(
    4, 1.0, 1.0, 0.0;
    nsweeps=1, maxdim=[10], cutoff=1e-8, use_gpu=false
)

# GPU representative call (only when CUDA is available)
if CUDA.has_cuda()
    SingleDMRGLMG.run_dmrg_lmg_single(
        4, 1.0, 1.0, 0.0;
        nsweeps=1, maxdim=[10], cutoff=1e-8, use_gpu=true
    )
end
