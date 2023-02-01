module Kdautoml

    using Random
    using Statistics
    using TOML
    using DelimitedFiles
    using LinearAlgebra
    using MacroTools
    using DataStructures
    using Combinatorics
    using CSV
    using DataFrames
    using Symbolics
    using SatisfiabilityInterface
    using AutoHashEquals
    using AbstractTrees
    using MultivariateStats
    using StatsBase
    using MutableNamedTuples
    using Tables
    using MLJ
    using MLJModelInterface
    import AbstractTrees: children, printnode
    import Base: push!, pop!
    
    export CodeNode,
           build,
           build_and_run_ml_pipeline,
           execute,
           paths,
           push!,
           AbstractComponent,
           AbstractState

    function __init__()
        # This bit is executed after module load
        # Declare container stuff
    end

    include("utils.jl")
    include("automaton.jl")
    include("kb.jl")
    include("sat.jl")
    include("program.jl")
    include("transition.jl")
    include("dfs/DeepFeatureSynthesis.jl")

end # module
