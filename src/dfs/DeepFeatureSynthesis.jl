"""
Simple run/test of module functionality
```
    features = Dict{Symbol, DeepFeatureSynthesis.AbstractFeature}()
    mg, _ = DeepFeatureSynthesis.generate_test_data()
    DeepFeatureSynthesis.deep_feature_synthesis(3, mg, [], features, 1, true);
    sort(vcat(collect(values(features))...), by=x->x.depth)
```
"""
module DeepFeatureSynthesis
    using LinearAlgebra
    using Statistics
    using TOML
    using AutoHashEquals
    using AbstractTrees
    using DataStructures
    using DataFrames
    using MacroTools
    using Graphs
    using MetaGraphs
    import ..MLJ
    using ..ControlFlow  # `kb_query` is defined in `ControlFlow`

    export AbstractFeature, deep_feature_synthesis

    include("features.jl")
    include("traversal.jl")

    # MLJ Interface
    mutable struct DeepFeatureSynthesisTransformer <: MLJ.Static
        path::AbstractString
        max_depth::Int
        calculate::Bool
    end

    function MLJ.transform(dfs::DeepFeatureSynthesisTransformer, _, data)
        kb = open(dfs.path) do io
            TOML.parse(io)
        end
        #kb = TOML.parse(open(dfs.path))
        features = deep_feature_synthesis(data, dfs.max_depth; kb=kb, calculate=dfs.calculate)
        df_features, _ = to_df(features)
        return df_features
    end

end
