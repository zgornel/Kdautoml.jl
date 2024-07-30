@reexport module KnowledgeBase

using ..TOML
using ..Random
using ..DelimitedFiles
using ..LinearAlgebra  # for `norm` in `rbf_kernel` (utils.jl)
using ..DataStructures
using ..MLJ
using ..DataFrames
using ..ConstraintSolver
using ..MacroTools
using ..Kdautoml  # needed for KB-defined precondition code in feature synthesis
                  # that when executed, references `Kdautoml`
import ..MLJModelInterface
import ..ControlFlow
import ..ProgramExecution

include("mlj_utils.jl")
include("neo4j.jl")
include("sat.jl")

export kb_load

const CS=ConstraintSolver
const model = CS.Model(CS.optimizer_with_attributes(CS.Optimizer,
                       "time_limit"=>1000,
                       "all_solutions"=>true,
                       "all_optimal_solutions"=>true))

const DEFAULT_PRECONDITION_SYMBOLS=(:AbstractPrecondition, :InputPrecondition, :DataPrecondition, :PipelinePrecondition)


function kb_load(filepath)
    TOML.parse(open(filepath))
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
