@reexport module KnowledgeSystem

using TOML                # used here
using DelimitedFiles      # for `kb_neo4j.jl`, `kb_native.jl`
using Reexport            # for KSSolver to use `@reexport`
using DataStructures      # used here
using Graphs
using MetaGraphs
import ..ControlFlow        # to extend API
import ..ProgramExecution   # to call API


# Utils that extent MLJ for use in KnowledgeBases
using MultivariateStats
using DataFrames
using MLJ
import MLJModelInterface
include("mlj_utils.jl")


# Knowledge Base API implementations
# Type definitions, exports
export AbstractKnowledgeBase
abstract type AbstractKnowledgeBase end
# Includes
include("kb_neo4j.jl")
include("kb_native.jl")


# Solver interface
function solve_csp end
function add_data_to_csp_solution end
include("solver.jl")


const DEFAULT_PRECONDITION_SYMBOLS=(:AbstractPrecondition, :InputPrecondition, :DataPrecondition, :PipelinePrecondition)


function ControlFlow.kb_load(filepath; kb_type=:neo4j, kb_flavour=:pipe_synthesis, connection=nothing)
    _connection = if kb_type == :neo4j && kb_flavour == :pipe_synthesis && connection == nothing
        (get_container(PIPESYNTHESIS_CONTAINER_NAME),
         get_neo4j_user(),
         get_neo4j_pass())
    elseif kb_type == :neo4j && kb_flavour == :feature_synthesis && connection == nothing
        (get_container(FEATURESYNTHESIS_CONTAINER_NAME),
         get_neo4j_user(),
         get_neo4j_pass())
    else
        connection
    end

    if kb_type==:neo4j
        return ControlFlow.kb_load(KnowledgeBaseNeo4j, filepath; connection=_connection)
    elseif kb_type==:native
        return ControlFlow.kb_load(KnowledgeBaseNative, filepath; connection=_connection)
    else
        @warn "Unrecognized knowledge base type $kb_type"
        return nothing
    end
end


function ControlFlow.kb_load(::Type{KnowledgeBaseNeo4j}, filepath; connection=nothing)
    KnowledgeBaseNeo4j(TOML.parse(open(filepath)), connection)
end


function ControlFlow.kb_load(::Type{KnowledgeBaseNative}, filepath; connection=nothing)
    KnowledgeBaseNative(TOML.parse(open(filepath)))  # connection is ignored
end


"""
Builds kb queries from input data structures, sends them to the kb and
fetches the results and processes them into response data structures
with which an constraint problem is built.
"""
function ControlFlow.kb_query(kb::K, ps_state::T) where {K<:AbstractKnowledgeBase, T<:Tuple{ControlFlow.AbstractComponent, Vector, Dict}}
    # Read state into variables
    component, pipeline, piperesults = ps_state

    # Look into component.metadata.preconditions and check what preconditions use
    # Precondition options:
    # • a tuple containing any of the values (:AbstractPrecondition, :InputPrecondition, :DataPrecondition, :PipelinePrecondition)
    # If the tuple is empty, all preconditions are used.
    allowed_preconditions = if hasproperty(component.metadata, :preconditions)
                                component.metadata.preconditions
                            else
                                DEFAULT_PRECONDITION_SYMBOLS
                            end

    # Read and parse components (including preconditions)
    component_name = ControlFlow.namify(component)
    kb_result = execute_kb_query(kb,
                          build_ps_query(component_name, K; allowed_preconditions);
                          output=true)

    # Build data structure for constraint solver
    datakb = build_ps_csp_data(kb_result, component_name, kb)

    # Do the same for preconditioned components, checking preconditions first
    state = (kb=kb, component=component, pipeline=pipeline, piperesults=piperesults)
    nm, s = solve_csp(datakb, state)
    sols = add_data_to_csp_solution(datakb, nm, s, Tuple(keys(datakb[:components])))
end


"""
Builds kb queries from input data structures, sends them to the kb and
fetches the results and processes them into response data structures
with which an constraint problem is built.

# Example for fs_state
```
fs_state = (read_table = :Subscriptions,
         write_table = :Subscriptions,
         input_feature = :id,
         input_eltype = Int64,
         use_vectors = true,
         input_data = 10×4 DataFrame
               Row │ id     service          customer  amount
                   │ Int64  Symbol           String    Float64
              ─────┼───────────────────────────────────────────
                 1 │     1  internet_10Mbps  sD6BY       19.99
                 2 │     2  internet_1Mbps   KwNTd        9.99
               ...
                10 │    10  internet_fiber   KwNTd       29.99,
         agg_column = nothing,
         mask_column = nothing)

```
"""
function ControlFlow.kb_query(kb::K, fs_state::NamedTuple) where K<:AbstractKnowledgeBase
    _to_string = ft -> last(split(string(ft), "."))
    kb_result = execute_kb_query(kb,
                    build_fs_query(_to_string(fs_state.feature_type), K);
                    output=true)

    datakb = build_fs_csp_data(kb_result, kb)

    nm, s = solve_csp(datakb, fs_state)
    sat_sols = add_data_to_csp_solution(datakb, nm, s, Tuple(keys(datakb[:components])))
    #combs = combinatorialize(compile_components(sat_sols, fs_state))
    combs = vcat([combinatorialize(compile_components(s, fs_state)) for s in sat_sols]...)
end


function compile_components(sol, fs_state)
    r = Vector{Pair{Symbol, Vector{Expr}}}()
    for (k, v) in sol  # k is component name, v in named tuple w. code, name, package etc.
        _f = eval(Meta.parse(strip(v.code)))
        _v = Base.invokelatest(_f, fs_state)  # _v isa Vector{Expr}
        push!(r, k => _v)
     end
    return r
end


function combinatorialize(v::Vector{Pair{T,S}}) where {T,S}
    n = mapreduce(p->length(p[2]), *, v)
    r = [Pair{T, eltype(S)}[] for _ in 1:n]
    for (i, (k,wi)) in enumerate(v)
        vals = repeat(wi, div(n, length(wi)))
        push!.(r, k.=>vals)
    end
    return r
end


function build_ps_csp_data(kb_response, component_name, kb)
    components = make_dict(kb_response)
    datakb = Dict(:components => MultiDict())
    for (comp, preconds) in components
        if haskey(get_recursively(kb, "resources/code/components"), comp)  # component is not abstract
            # gather precondition code and arguments
            preconditions = []
            if !isempty(preconds)
                for precond in preconds
                    _pdata = get_recursively(kb, "ontology/preconditions/$precond")
                    func_name, func_args = _pdata["function"], _pdata["function_args"]
                    push!(preconditions, (name=precond,
                                          code=get_recursively(kb, "resources/code/preconditions/$func_name/code"),
                                          package=get_recursively(kb, "resources/code/preconditions/$func_name/package"),
                                          args=func_args))
                end
            end
            _data = (name=comp,  # concrete component, not abstract
                     code=get_recursively(kb, "resources/code/components/$comp/code"),
                     hyperparameters=get_recursively(kb, "resources/code/components/$comp/hyperparameters"),
                     package=get_recursively(kb, "resources/code/components/$comp/package"),
                     preconditions=preconditions)
            push!(datakb[:components], Symbol(component_name)=>_data)
        end
    end
    return datakb
end


function build_fs_csp_data(kb_response, kb)
    component_data = MultiDict()
    for (acomp, dcomp, precond) in eachrow(kb_response)
        # Build precondition
        precondition = nothing
        precond_name = replace(strip(precond), "\""=>"")
        dcomp_name = replace(strip(dcomp), "\""=>"")
        if !isempty(precond_name)
                _pdata = get_recursively(kb, "ontology/preconditions/$precond_name")
                func_name, func_args = _pdata["function"], _pdata["function_args"]
                precondition = (name=precond_name,
                                code=get_recursively(kb, "resources/code/preconditions/$func_name/code"),
                                args=func_args)
        end

        # Find discrete components and add preconditions to them or add new entry
        acomp_symb = Symbol(acomp)
        pos = findall(v->hasproperty(v, :name)&&v.name==dcomp_name, get(component_data, acomp_symb, []))
        if isempty(pos)  # add new entry
            component = (name=dcomp_name,
                         code=get_recursively(kb, "resources/code/components/$dcomp_name/code"),
                         hyperparameters=get_recursively(kb, "resources/code/components/$dcomp_name/hyperparameters"),
                         package=get_recursively(kb, "resources/code/components/$dcomp_name/package"),
                         preconditions=!isnothing(precondition) ? Any[precondition] : Any[])
            push!(component_data, acomp_symb=>component)
        else  # just add precondition
            for p in pos
                !isnothing(precondition) && push!(component_data[acomp_symb][p].preconditions, precondition)
            end
        end
    end
    datakb = Dict(:components=>component_data)
    return datakb
end


function get_recursively(kb::K, args...; kwargs...) where K<:AbstractKnowledgeBase
    return get_recursively(kb.data, args...;kwargs...)
end

function get_recursively(d::Dict, ks; splitter='/', default=nothing)
    keys = split(strip(isequal(splitter), ks), string(splitter))
    reduce((d, k)->get(d, k, default), keys, init=d)
end


# Creates a MultiDict from a neo4j matrix result
function make_dict(m)
    d= MultiDict{String, String}();
    for r in eachrow(m)
        pcond = replace(strip(r[2]), r"\""=>"")
        if isempty(pcond)
            !haskey(d, r[1]) && (push!(d, r[1]=>"");
            pop!(d[r[1]]))  # remove empty string, keep key in MultiDict
        else
            push!(d, r[1]=>pcond)
        end
    end
    return d
end


__get_data(state) = begin
    _, pipeline, piperesults = state
    for i in length(pipeline):-1:1
       res = (piperesults[(pipeline[i]).id])
       !isnothing(res) && return res.pipe_out  # returns the last non-nothing pipeline ouput
    end
end

ControlFlow.get_update_nodes(kbnodes, ps_state::Tuple{ControlFlow.AbstractComponent, Vector, Dict}) = begin
    component = ps_state[1]
    arguments = hasproperty(component.metadata, :arguments) ? component.metadata.arguments : ()
    updatenodes = ProgramExecution.CodeNode[]
    if !isempty(Iterators.flatten(kbnodes))
        for n in kbnodes
            _, nodedata = n[1]        #TODO: Adapt this to pick up several components if necessary;
                                      #      Here it is assumed that a single node is returned all the time (no combinatorial)
            push!(updatenodes, ProgramExecution.CodeNode(nodedata.name, (code=nodedata.code, hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=arguments), ProgramExecution.CodeNode[]))
        end
    end
    return updatenodes
end

ControlFlow.get_update_nodes(kbnodes, ps_state::Tuple{ControlFlow.FeatureSelection, Vector, Dict}) = begin
    component = ps_state[1]
    @assert hasproperty(component.metadata, :arguments) "Missing arguments "
    @assert length(kbnodes) == 1 "Feature selection should return only one feasible way of obtaining feature subsets"
    if component.metadata.arguments[1] == "random"
        c = ControlFlow.RandomFeatureSelection((arguments=component.metadata.arguments[2:end], execute=component.metadata.execute))
        return ControlFlow.get_update_nodes(kbnodes, (c, ps_state[2:end]...))
    elseif component.metadata.arguments[1] == "direct"
        c = ControlFlow.DirectFeatureSelection((arguments=component.metadata.arguments[2], execute=component.metadata.execute))
        return ControlFlow.get_update_nodes(kbnodes, (c, ps_state[2:end]...))
    else
        @error "Unknown FeatureSelection argument format."
    end
end

ControlFlow.get_update_nodes(kbnodes, ps_state::Tuple{ControlFlow.RandomFeatureSelection, Vector, Dict}) = begin
    component = ps_state[1]
    data = __get_data(ps_state)
    kbnode = kbnodes[1][1]
    ns, nf = component.metadata.arguments  # number of subsets and number of features/subset
    _, m = size(data)
    nss = ifelse(nf >= m, 1, ns)  #TODO: use combinatorics for better assesment of max number of feature subsets
    feature_subsets = unique(sort.([sample(1:m, nf, replace=false) for _ in 1:nss]))
    updatenodes = ProgramExecution.CodeNode[]
    for (i, fs) in enumerate(feature_subsets)
        _, nodedata = kbnode
        push!(updatenodes, ProgramExecution.CodeNode("$(nodedata.name)=>$fs", (code=nodedata.code,  hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=(fs,)), ProgramExecution.CodeNode[]))
    end
    return updatenodes
end

ControlFlow.get_update_nodes(kbnodes, ps_state::Tuple{ControlFlow.DirectFeatureSelection, Vector, Dict}) = begin
    component = ps_state[1]
    data = __get_data(ps_state)
    kbnode = kbnodes[1][1]
    feature_subsets = component.metadata.arguments
    m, _ = size(data)
    map(fs->filter!(in(1:m), fs), feature_subsets)
    updatenodes = ProgramExecution.CodeNode[]
    for (i, fs) in enumerate(feature_subsets)
        _, nodedata = kbnode
        push!(updatenodes, ProgramExecution.CodeNode("$(nodedata.name)=>$fs", (code=nodedata.code, hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=(fs,)), ProgramExecution.CodeNode[]))
    end
    return updatenodes
end

end # module
