struct KnowledgeBaseNative <: AbstractKnowledgeBase
    data::Dict
    graph::MetaDiGraph
end

Base.show(io::IO, kb::KnowledgeBaseNative) = begin
    mb_size = Base.summarysize(kb.data)/(1024^2)
    print(io, "KnowledgeBaseNative, $mb_size MB of data, $(Graphs.nv(kb.graph)) nodes, $(Graphs.ne(kb.graph)) links")
end

KnowledgeBaseNative(data) = KnowledgeBaseNative(data, __build_metagraph(data))


abstract type AbstractQuery end

struct PipelineSynthesisQuery <: AbstractQuery
    node_label::String
    allowed_preconditions::Tuple   # tuple of symbols with allowed precondition types
end

struct FeatureSynthesisQuery <: AbstractQuery
    node_label::String
end


# Actual implementation
function build_ps_query(node_label::String,
                        ::Type{KnowledgeBaseNative};
                        allowed_preconditions=DEFAULT_PRECONDITION_SYMBOLS)
    return PipelineSynthesisQuery(node_label, allowed_preconditions)
end

function build_fs_query(feature_type, ::Type{KnowledgeBaseNative})
    return FeatureSynthesisQuery(node_label)
end


_get_node_label(kb, id) = props(kb.graph, id)[:label]

_get_node_type(kb, id) = props(kb.graph, id)[:type]

_get_node_id_fromlabel(kb, label) = kb.graph[label, :label]

function _get_edge_type(kb, edge)
    dd = props(kb.graph, edge)
    if haskey(dd, :type)
        return dd[:type]
    else
        return nothing
    end
end

function _get_subtree(kb, id; connect_f=Graphs.inneighbors)
    subtree = [id]
    front = connect_f(kb.graph, id)
    while true
        if isempty(front)
            break
        end
        v = pop!(front)
        if !in(v, subtree)
            push!(subtree, v)
        end
        #TODO: See if its worth filtering by edge type (i.e. :ISA)
        new_front = setdiff(connect_f(kb.graph, v), front)
        append!(front, new_front)
    end
    return subtree
end


function execute_kb_query(kb::KnowledgeBaseNative, query::PipelineSynthesisQuery; kwargs...)
    # Find node id
    _nodes = collect(filter_vertices(kb.graph, :label, query.node_label))
    if length(_nodes) > 1 "ode label is not unique in the KB"
        @error "Node label \"$(query.node_label)\" is not unique in KB"
    elseif length(_nodes) == 0
        @error "No nodes with label \"$(query.node_label)\" found in KB"
    end
    node_id = first(_nodes)

    # Find subtree/subgraph of connected nodes
    subtree = _get_subtree(kb, node_id; connect_f=Graphs.inneighbors)

    # Look for directly linked preconditions and add them
    node_pcond_links = []
    for node in subtree
        has_pconds = false
        for l_node in outneighbors(kb.graph, node)  # loop over all outbound
            if _get_node_type(kb, l_node) == :precondition
                has_pconds = true
                push!(node_pcond_links, (_get_node_label(kb, node) => _get_node_label(kb, l_node)))
            end
        end
        !has_pconds && push!(node_pcond_links, (_get_node_label(kb, node) => ""))  # default, no preconditions
    end

    # Filter by precondition
    # TODO: Fix incomplete precondition list bug!
    out = []
    for (node, pcond) in node_pcond_links
        if pcond == ""
            push!(out, node => pcond)
        else
            push!(out, node => "")
            node_id = _get_node_id_fromlabel(kb, node)
            pcond_id = _get_node_id_fromlabel(kb, pcond)
            subtree = _get_subtree(kb, _get_node_id_fromlabel(kb, pcond), connect_f=Graphs.outneighbors)
            for p in subtree
                if pcond_id!== p && _get_edge_type(kb, Edge(pcond_id, p)) == :ISA &&
                   _get_node_label(kb, p) in string.(query.allowed_preconditions)
                        push!(out, node => pcond)
                end
            end
        end
    end
    unique!(out)
    out_matrix = Matrix{String}(undef, length(out), 2)
    for (i, (node, pcond)) in enumerate(out)
        out_matrix[i,:] .= node, " \"$pcond\""
    end
    return out_matrix
end


function execute_kb_query(kb::KnowledgeBaseNative, query::FeatureSynthesisQuery; kwargs...)
end


# Returns a vector of statements that can be ran by execute_kb_query
function __build_metagraph(kb)

    # Add a component to the metagraph with a specific label
    function add_pipe_component!(mg, node_label, node_type)
        add_vertices!(mg, 1);
        #TODO: Decide whether to add more data here i.e. code etc.
        set_props!(mg, nv(mg), Dict(:type=>node_type, :label=>node_label))
    end

    function add_relation!(mg, src, dst, rel, index_on)
        src_idx = mg[src, index_on]
        dst_idx = mg[dst, index_on]
        add_edge!(mg, src_idx, dst_idx)
        set_prop!(mg, src_idx, dst_idx, :type, Symbol(rel))
    end

    mg = MetaDiGraph()
    index_on = :label
    set_indexing_prop!(mg, index_on)

    # Create component nodes
    component_nodes = kb["ontology"]["components"]
    for node in keys(component_nodes)
        add_pipe_component!(mg, node, :component)
    end

    # Create precondition nodes
    precondition_nodes = kb["ontology"]["preconditions"]
    for node in keys(precondition_nodes)
        add_pipe_component!(mg, node, :precondition)
    end

    # Create relations
    relations = kb["ontology"]["relations"]
    for (relation, relargs) in relations
        for (src, dst) in relargs["data"]
            add_relation!(mg, src, dst, relation, index_on)
        end
    end
    return mg
end
