@auto_hash_equals struct TableLink  # a link from src=(:table, :col) --(to)--> dst=(:table, :col)
    src::Tuple{Symbol, Symbol}
    dst::Tuple{Symbol, Symbol}
end

abstract type AbstractFeature{T} end

@auto_hash_equals mutable struct ReverseFeature{T} <: AbstractFeature{T}
    table::Symbol
    ast::Expr
    values::Union{Nothing, Vector{T}}
end

@auto_hash_equals mutable struct DirectFeature{T} <: AbstractFeature{T}
    table::Symbol
    ast::Expr
    values::Union{Nothing, Vector{T}}
end

@auto_hash_equals mutable struct EntityFeature{T} <: AbstractFeature{T}
    table::Symbol
    ast::Expr
    values::Union{Nothing, Vector{T}}
end

@auto_hash_equals mutable struct IdentityFeature{T} <: AbstractFeature{T}
    table::Symbol
    ast::Expr
    values::Union{Nothing, Vector{T}}
end

Base.eltype(feature::Type{<:AbstractFeature{T}}) where T = T
Base.size(feature::AbstractFeature, idx...) = size(feature.values, idx...)

# The base AST for features looks something like this
#              
#             (f)                 ----> `(f)` corresponds to `ScalarReducer`
#              |
#          .-------.
#          |       |              ----> `(g)` corresponds to `TensorReducer`
#         (xl)     (g)            ----> `(xl)` is obstained by calling `ScalarGetter(data)`              
#                  |
#               (filter)          ----> `(filter)` corresponds to `TensorFilter`
#                  |
#              .-------.
#              |       |
#             (Xl)   (condition)  ----> `(Xl)` is obtained using `TensorGetter(data)`
#                     / \         ----> (`condition`) corresponds to `TensorConditioner` (must be a logical combination of functions/functors)
#                    /   \
#                 (xk)   (Xk)     ----> `(condition)` can rely on (`xk`) and (`Xk`) to build the condition function which will select into `Xl`
#
#

Base.@kwdef struct FeatureComponents
    ScalarReducer::Expr = :(x->x)
    ScalarGetter::Expr = :(x->x)
    TensorReducer::Expr = :(x->x)
    TensorConditioner::Expr = :(x->true)
    TensorFilter::Expr = :(v->filter(x->true, v))
    TensorGetter::Union{Nothing,Expr} = nothing
end

const DATA_SYMBOL = :data
const IDX_SYMBOL = :idx


#I-FEATURES
function build_generic_feature_ast(::Type{IdentityFeature}; use_vectors=false, data_symbol=DATA_SYMBOL, idx_symbol=IDX_SYMBOL)
        return quote
            ($data_symbol, $idx_symbol)->(x->x)(ScalarGetter($data_symbol, $idx_symbol))
            end
end

#E-FEATURES
function build_generic_feature_ast(::Type{EntityFeature}; use_vectors=false, data_symbol=DATA_SYMBOL, idx_symbol=IDX_SYMBOL)
    if !use_vectors
        return quote
                ($data_symbol, $idx_symbol)->ScalarReducer(ScalarGetter($data_symbol, $idx_symbol))
            end
    else
        return quote
                ($data_symbol, $idx_symbol)->ScalarReducer(ScalarGetter($data_symbol, $idx_symbol), TensorReducer(TensorFilter(TensorConditioner, TensorGetter($data_symbol, $idx_symbol))))
            end
    end
end

# D-FEATURES: these are obtained just by copying R-FEATURES
function build_generic_feature_ast(::Type{DirectFeature}; use_vectors=false, data_symbol=DATA_SYMBOL, idx_symbol=IDX_SYMBOL)
    return quote
            ($data_symbol, $idx_symbol)->first(getindex(TensorGetter($data_symbol, $idx_symbol), TensorConditioner))
        end
end

# R-FEATURES
function build_generic_feature_ast(::Type{ReverseFeature}; use_vectors=false, data_symbol=DATA_SYMBOL, idx_symbol=IDX_SYMBOL)
    return quote
            ($data_symbol, $idx_symbol)->ScalarReducer(ScalarGetter($data_symbol, $idx_symbol), TensorReducer(getindex(TensorGetter($data_symbol, $idx_symbol), TensorConditioner)))
        end
end

const FUNC_NODES = (:ScalarReducer, :ScalarGetter, :TensorReducer, :TensorFilter, :TensorGetter)
const OTHER_NODES = (:TensorConditioner,)

function ast_strip_col(f_ast, col)
    pattern = :(df.$col)
    replacement = :(df.COLUMN)
    MacroTools.postwalk(f_ast) do x
        if @capture(x, p_)
            p == pattern ? replacement : p
        end
    end
end


# Build an actual compilable feature function using an AST and a V
function materialize(ast::Expr, comps::FeatureComponents)
    MacroTools.postwalk(ast) do x
        if @capture(x, f_(xs__))
            if f in FUNC_NODES
                return :(($(getproperty(comps, f))($(xs...))))
            else
                return x
            end
        elseif @capture(x, f_)
            if f in OTHER_NODES
                return :(($(getproperty(comps, f))))
            else
                return x
            end
        else
            return x
        end
    end
end
