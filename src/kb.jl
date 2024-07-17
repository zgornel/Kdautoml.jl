@reexport module KnowledgeBase

using ..TOML
using ..Random
using ..DelimitedFiles
using ..LinearAlgebra  # for `norm` in `rbf_kernel` (utils.jl)
using ..DataStructures
using ..MLJ
using ..DataFrames
using ..Symbolics
using ..ConstraintSolver
using ..MacroTools
using ..Kdautoml  # needed for KB-defined precondition code in feature synthesis
                  # that when executed, references `Kdautoml`
import ..MLJModelInterface
import ..ControlFlow
import ..ProgramExecution

export kb_load,
       get_container, get_neo4j_user, get_neo4j_pass,
       kb_to_neo4j_statements, cypher_shell,
       PIPESYNTHESIS_CONTAINER_NAME, FEATURESYNTHESIS_CONTAINER_NAME,
       build_and_run_ml_pipeline,
       df2vec, xor_linear_separability,
       FeatureProduct, Kernelizer, rbf_kernel,
       change_into_UnivariateFinite, build_ranges

const CS=ConstraintSolver
const model = CS.Model(CS.optimizer_with_attributes(CS.Optimizer,
                     "time_limit"=>1000,
                     "all_solutions"=>true,
                     "all_optimal_solutions"=>true))
include("utils.jl")
include("sat.jl")

const DEFAULT_PRECONDITION_SYMBOLS=(:AbstractPrecondition, :InputPrecondition, :DataPrecondition, :PipelinePrecondition)

function kb_load(filepath)
    TOML.parse(open(filepath))
end


# Returns a vector of statements that can be ran by cypher_shell
function kb_to_neo4j_statements(kb)
    STMTs = String[]

    # Template statement creation
    component_node_template(node_label) = "CREATE (n:$node_label)"
    #TODO(Corneliu): Check whether it makes sense to add the function code and args
    # in the neo4j db or not; at this point type to string conversion is difficult
    # and potentially not needed.
    # '0c3586a' is the latest commit supporting precondition func+args in db
    precondition_node_template(node_label, func, args) = "CREATE (n:$node_label)"
    relation_template(src, dst, rel) = "MATCH (a:$src), (b:$dst) CREATE (a)-[r:$rel]->(b)"

    # Create component nodes
    component_nodes = kb["ontology"]["components"]
    for node in keys(component_nodes)
        push!(STMTs, component_node_template(node))
    end

    # Create precondition nodes
    precondition_nodes = kb["ontology"]["preconditions"]
    for node in keys(precondition_nodes)
        if haskey(precondition_nodes[node], "function")
            # Preconditions with callback
            push!(STMTs, precondition_node_template(node,
                precondition_nodes[node]["function"],
                string(precondition_nodes[node]["function_args"])))
        else
            # Preconditions without callback
            push!(STMTs, component_node_template(node))
        end
    end

    # Create relations
    relations = kb["ontology"]["relations"]
    for (relation, relargs) in relations
        for (src, dst) in relargs["data"]
            push!(STMTs, relation_template(src, dst, relation))
        end
    end
    return STMTs
end


"""
Builds kb queries from input data structures, sends them to the kb and
fetches the results and processes them into response data structures
with which an constraint problem is built.
"""
function ControlFlow.kb_query(kb, ps_state::T; connection=nothing) where {T<:Tuple{ControlFlow.AbstractComponent, Vector, Dict}}

    # Read state into variables
    component, pipeline, piperesults = ps_state

    # Creates a MultiDict from a neo4j matrix result
    make_dict(m) = begin
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

    # Look into component.metadata.preconditions and check what preconditions use
    # Precondition options:
    # • a tuple containing any of the values (:AbstractPrecondition, :InputPrecondition, :DataPrecondition, :PipelinePrecondition)
    # If the tuple is empty, all preconditions are used.

    # Functions that build queries
    #   Note: the -[:LINK*0..]- return nodes 0 or more LINKs away. Useful to return target node along with linked nodes
    precondition_symbols = if hasproperty(component.metadata, :preconditions)
                               component.metadata.preconditions
                           else
                               DEFAULT_PRECONDITION_SYMBOLS
                           end
    function f_query_components(action; precondition_symbols=DEFAULT_PRECONDITION_SYMBOLS)
        symb_p = "p"
        where_parts = ["($symb_p)-[:ISA]->(:$p)" for p in precondition_symbols]
        where_clause = "WHERE " * join(where_parts, " OR ")
        query = """
            // list of nodes selected by specific preconditions
            MATCH ($symb_p)<-[:PRECONDITIONED_BY]-(n)-[:ISA*0..]->(:$action)
            $where_clause
            UNWIND labels(n) AS nl UNWIND labels($symb_p) AS $(symb_p)l
            RETURN nl, $(symb_p)l
            // list of all nodes (equivalent to rest of preconditions ignored i.e. set to true)
            UNION
            MATCH (n)-[:ISA*0..]->(:$action)
            UNWIND labels(n) AS nl unwind "" as $(symb_p)l
            RETURN nl, $(symb_p)l
            """
       return query
    end

    # Read and parse components (including preconditions)
    component_name = ControlFlow.namify(component)
    connection == nothing && (connection = (get_container(PIPESYNTHESIS_CONTAINER_NAME),
                                            get_neo4j_user(),
                                            get_neo4j_pass())
                             )
    _r = cypher_shell(connection...,
                      f_query_components(component_name; precondition_symbols);
                      output=true)
    components = make_dict(parse_neo4j_result(_r))

    # Build data structure for constraint solver
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
function ControlFlow.kb_query(kb, fs_state::NamedTuple; connection=nothing)
    f_query_components(feature_type) = """
        MATCH (n:$feature_type)-[:HASA]->(ac)<-[:ISA*]-(c)-[:PRECONDITIONED_BY]->(p)
        UNWIND labels(ac) as acl UNWIND labels(c) as cl UNWIND labels(p) as pl
        RETURN acl, cl, pl
        UNION
        MATCH (n:$feature_type)-[:HASA]->(ac)<-[:ISA*]-(c) WHERE NOT (c)<-[:PRECONDITIONED_BY]->()
        UNWIND labels(ac) as acl UNWIND labels(c) as cl UNWIND "" as pl
        RETURN acl, cl, pl
        """

    _to_string = ft -> last(split(string(ft), "."))
    connection == nothing && (connection = (get_container(FEATURESYNTHESIS_CONTAINER_NAME),
                                            get_neo4j_user(),
                                            get_neo4j_pass())
                             )
    _r = cypher_shell(connection...,
                      f_query_components(_to_string(fs_state.feature_type));
                      output=true)
    _r = parse_neo4j_result(_r)

    component_data = MultiDict()
    for (acomp, dcomp, precond) in eachrow(_r)
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
    for n in kbnodes
        _, nodedata = n[1]        #TODO: Adapt this to pick up several components if necessary;
                                  #      Here it is assumed that a single node is returned all the time (no combinatorial)
        push!(updatenodes, ProgramExecution.CodeNode(nodedata.name, (code=nodedata.code, hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=arguments), ProgramExecution.CodeNode[]))
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
