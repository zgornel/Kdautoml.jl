

@register_symbolic ∨(x, y) false   # \wedge
@register_symbolic ∧(x, y) false   # \vee
@register_symbolic ¬(x) false      # \neg

# @register_symbolic ∨(x...)
# @register_symbolic ∨(x::AbstractVector{T} where T)

const DiscreteTypes = Union{Integer, Symbol}

#@register_symbolic Base.:∈(x::Num, y::AbstractRange{T} where T<:DiscreteTypes) false
#@register_symbolic Base.:∈(x::Num, y::UnitRange{T} where T<:DiscreteTypes) false  # false prevents defining overloads
#@register_symbolic Base.:∈(x::Num, y::AbstractVector{T} where T<:DiscreteTypes) false
#@register_symbolic Base.:∈(x::Num, y::Vector{T} where T<:DiscreteTypes) false
#@register_symbolic Base.:∈(x::Num, y::Tuple) false

#Backport from https://github.com/dpsanders/SatisfiabilityInterface.jl/pull/37
@register_symbolic Base.:∈(x::Real, y::UnitRange) false  # false prevents defining overloads
@register_symbolic Base.:∈(x::Real, y::Vector{Symbol}) false  # false prevents defining overloads
@register_symbolic Base.:∈(x::Real, y::Vector{Int}) false  # false prevents defining overloads
# TODO: Varargs ∨


∨(x::Vector{T}) where {T} = isempty(x) ? T[] : ∨(x...)


∨(x...) = foldl(∨, x)
# ∨(x...) = Symbolics.Term(∨, [x...])
∧(x...) = foldl(∧, x)

