module Kdautoml
    using Random
    using Statistics
    using TOML
    using DelimitedFiles
    using LinearAlgebra
    using Reexport
    using MacroTools
    using DataStructures
    using Combinatorics
    using CSV
    using DataFrames
    using ConstraintSolver
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

    export DeepFeatureSynthesis,
           AbstractComponent,
           AbstractState,
           CodeNode,
           build,
           build_and_run_ml_pipeline,
           execute,
           paths,
           push!

    function __init__()
        # This bit is executed after module load
        # Declare container stuff
    end

    include("control.jl")     # CF - highest level, most abstract, has all top-level definitions
    include("program.jl")     # PE - second level, needs stuff defined previously
    include("ks.jl")          # KS - third level, needs methods defined in the previous two
    include("dfs/DeepFeatureSynthesis.jl") # second level also, defines KB query interface for features

end # module
