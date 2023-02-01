# FSM transition wrapper: implements communication between
# control flow with the kb and the program representation
function transition(state::AbstractState,
                    component::AbstractComponent;
                    kb=nothing,
                    pipelines=nothing,
                    printbuffer="__tree__")
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
        kb_result = kb_query(kb, ps_state)

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


__get_data(state) = begin
    _, pipeline, piperesults = state
    for i in length(pipeline):-1:1
       res = (piperesults[(pipeline[i]).id])
       !isnothing(res) && return res.pipe_out  # returns the last non-nothing pipeline ouput
    end
end

get_update_nodes(kbnodes, ps_state::Tuple{AbstractComponent, Vector, Dict}) = begin
    component = ps_state[1]
    arguments = hasproperty(component.metadata, :arguments) ? component.metadata.arguments : ()
    updatenodes = CodeNode[]
    for n in kbnodes
        _, nodedata = n[1]        #TODO: Adapt this to pick up several components if necessary;
                                  #      Here it is assumed that a single node is returned all the time (no combinatorial)
        push!(updatenodes, CodeNode(nodedata.name, (code=nodedata.code, hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=arguments), CodeNode[]))
    end
    return updatenodes
end

get_update_nodes(kbnodes, ps_state::Tuple{FeatureSelection, Vector, Dict}) = begin
    component = ps_state[1]
    @assert hasproperty(component.metadata, :arguments) "Missing arguments "
    @assert length(kbnodes) == 1 "Feature selection should return only one feasible way of obtaining feature subsets"
    if component.metadata.arguments[1] == "random"
        c = RandomFeatureSelection((arguments=component.metadata.arguments[2:end], execute=component.metadata.execute))
        return get_update_nodes(kbnodes, (c, ps_state[2:end]...))
    elseif component.metadata.arguments[1] == "direct"
        c = DirectFeatureSelection((arguments=component.metadata.arguments[2], execute=component.metadata.execute))
        return get_update_nodes(kbnodes, (c, ps_state[2:end]...))
    else
        @error "Unknown FeatureSelection argument format."
    end
end

get_update_nodes(kbnodes, ps_state::Tuple{RandomFeatureSelection, Vector, Dict}) = begin
    component = ps_state[1]
    data = __get_data(ps_state)
    kbnode = kbnodes[1][1]
    ns, nf = component.metadata.arguments  # number of subsets and number of features/subset
    _, m = size(data)
    nss = ifelse(nf >= m, 1, ns)  #TODO: use combinatorics for better assesment of max number of feature subsets
    feature_subsets = unique(sort.([sample(1:m, nf, replace=false) for _ in 1:nss]))
    updatenodes = CodeNode[]
    for (i, fs) in enumerate(feature_subsets)
        _, nodedata = kbnode
        push!(updatenodes, CodeNode("$(nodedata.name)=>$fs", (code=nodedata.code,  hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=(fs,)), CodeNode[]))
    end
    return updatenodes
end

get_update_nodes(kbnodes, ps_state::Tuple{DirectFeatureSelection, Vector, Dict}) = begin
    component = ps_state[1]
    data = __get_data(ps_state)
    kbnode = kbnodes[1][1]
    feature_subsets = component.metadata.arguments
    m, _ = size(data)
    map(fs->filter!(in(1:m), fs), feature_subsets)
    updatenodes = CodeNode[]
    for (i, fs) in enumerate(feature_subsets)
        _, nodedata = kbnode
        push!(updatenodes, CodeNode("$(nodedata.name)=>$fs", (code=nodedata.code, hyperparameters=nodedata.hyperparameters, package=nodedata.package, arguments=(fs,)), CodeNode[]))
    end
    return updatenodes
end


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
