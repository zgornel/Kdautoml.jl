@reexport module ControlFlow

using ..AbstractTrees

export paths, prune!, build, execute, # PE interface
       kb_query, # KB interface
       # top-level symbols (imported by KB)
       transition, namify,
       FSMTransitionError, AbstractState, NoData, Data, ModelableData, Model, End,
       AbstractComponent, LoadData, PreprocessData, SplitData, SPlitHoldOut, SplitCV,
       SplitStratifiedCV, FeatureOperation, DimensionalityReduction, DFS, ProductFeatures,
       FeatureSelection, RandomFeatureSelection, DirectFeatureSelection, SelectModel, ModelData,
       EvalModel, EvalClassifier


# ProgramExecution Interface
# i.e. functions that need to be have methods in the `ProgramExecution` module
function paths end
function prune! end
function build end
function execute end


# KnowledgeBase interface
# i.e. functions that need to be have methods in the `KnowledgeBase` module
function kb_query end
function get_update_nodes end


# Definitions for states, inputs and the FSM
include("automaton.jl")

# FSM transition wrapper: implements communication between
# control flow with the kb and the program representation
function transition(state::AbstractState,
                    component::AbstractComponent;
                    kb=nothing,
                    pipelines=nothing,
                    printbuffer="__tree__",
                    connection=nothing)
    if kb == nothing || pipelines == nothing
        @error "`kb` and `pipelines` need to be provided"
    end

    # Compute already next state (returning marks the actual transition)
    next_state = _transition(state, component)
    @info "FSM: state=$(namify(state)), input=$(namify(component)) --> $(namify(next_state))"

    # Loop through pipelines of available pipelines
    root = pipelines.tree
    current_leaves = map(n->n.id, collect(Leaves(root)))
    for (leaf, treepath) in paths(root, current_leaves)
        @info "FSM:\tQuerying for an `$(namify(component))` for pipeline=$(make_pipe_text(treepath))"

        # create query - sends name of pipe leaf (last component) to query creation
        ps_state = (component, treepath, pipelines.artifacts)
        kb_result = kb_query(kb, ps_state; connection)

        # Update pipelines
        updatenodes = get_update_nodes(kb_result, ps_state)
        if !isempty(updatenodes)
            @info "FSM:\tReceived from KB $(length(updatenodes)) component(s) of type `$(namify(component))` for pipeline=$(make_pipe_text(treepath))"
            append!(leaf.children, updatenodes) # add code nodes to the tree
        else
            @info "FSM:\tPruning pipeline=$(make_pipe_text(treepath))"
            popped = prune!(root, leaf)  # if one cannot add to the current pipe anymore, it is pruned
            for id in getproperty.(popped, :id)
                delete!(pipelines.artifacts, id)
            end
        end

        # Print pipelines
        printbuffer!= nothing && open(printbuffer, "w") do io
            AbstractTrees.print_tree(io, root, maxdepth=100)
            flush(io)
        end

        # Execute pipelines
        for node in updatenodes
            program = build(vcat(treepath, node), pipelines)
            if hasproperty(component.metadata, :execute) && component.metadata.execute
                pipelines.artifacts[node.id] = execute_with_indicator(root, node, program; buffer=printbuffer)
            end
        end
    end
    return next_state
end


make_pipe_text(treepath) = join(map(x->x.name, treepath), " -> ")


function execute_with_indicator(tree, node, program; buffer=nothing)
    if buffer != nothing
        node.name=">>$(node.name)<<"
        open(buffer, "w") do io
            AbstractTrees.print_tree(io, tree, maxdepth=100)
            flush(io)
        end
    end
    # Actual execution
    result = execute(program)
    ###
    if buffer != nothing
        node.name = replace(node.name, r"(<|>)"=>"")
        open(buffer, "w") do io
            AbstractTrees.print_tree(io, tree, maxdepth=100)
            flush(io)
        end
    end
    return result
end


# Useful function
namify(sometype) = String(typeof(sometype).name.name)


end # module
