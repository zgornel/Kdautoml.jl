export PIPESYNTHESIS_CONTAINER_NAME, FEATURESYNTHESIS_CONTAINER_NAME,
       get_neo4j_user, get_neo4j_pass, get_container_name


PIPESYNTHESIS_CONTAINER_NAME = "neo4j_pipesynthesis_kb"
FEATURESYNTHESIS_CONTAINER_NAME = "neo4j_featuresynthesis_kb"


# TODO: Make this parametric
get_neo4j_user() = get(ENV, "NEO4J_USER", "neo4j")
get_neo4j_pass() = get(ENV, "NEO4J_PASS", "test")
get_container(container_name) = chop(read(pipeline(`docker ps `,`grep "$container_name"`, `awk '{print $1}'`), String))


struct KnowledgeBaseNeo4j <: AbstractKnowledgeBase
    data::Dict
    connection::Tuple
end

Base.show(io::IO, kb::KnowledgeBaseNeo4j) = begin
    container, user, pass = kb.connection
    mb_size = Base.summarysize(kb.data)/(1024^2)
    print(io, "KnowledgeBase (Neo4j: $user:****@$container), $mb_size MB of data")
end

# Returns a vector of statements that can be ran by execute_kb_query
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
    component_nodes = kb.data["ontology"]["components"]
    for node in keys(component_nodes)
        push!(STMTs, component_node_template(node))
    end

    # Create precondition nodes
    precondition_nodes = kb.data["ontology"]["preconditions"]
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
    relations = kb.data["ontology"]["relations"]
    for (relation, relargs) in relations
        for (src, dst) in relargs["data"]
            push!(STMTs, relation_template(src, dst, relation))
        end
    end
    return STMTs
end


# Functions that build queries
#   Note: the -[:LINK*0..]- return nodes 0 or more LINKs away. Useful to return target node along with linked nodes
function build_ps_query(action, kb::Type{KnowledgeBaseNeo4j}; precondition_symbols=DEFAULT_PRECONDITION_SYMBOLS)
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


function build_fs_query(feature_type, kb::Type{KnowledgeBaseNeo4j})
    return """
    MATCH (n:$feature_type)-[:HASA]->(ac)<-[:ISA*]-(c)-[:PRECONDITIONED_BY]->(p)
    UNWIND labels(ac) as acl UNWIND labels(c) as cl UNWIND labels(p) as pl
    RETURN acl, cl, pl
    UNION
    MATCH (n:$feature_type)-[:HASA]->(ac)<-[:ISA*]-(c) WHERE NOT (c)<-[:PRECONDITIONED_BY]->()
    UNWIND labels(ac) as acl UNWIND labels(c) as cl UNWIND "" as pl
    RETURN acl, cl, pl
    """
end


#TODO: Check docker is installed, container valid, etc
function execute_kb_query(kb::KnowledgeBaseNeo4j, cypher_cmd; wait=true, output=false)
    # Function that parses neo4j results into a Matrix{String}
    function parse_neo4j_result(result)
        if !isempty(result)
            return String.(readdlm(IOBuffer(result), ',','\n';header=true)[1])
        else
            return String[]
        end
    end

    # Read parameters
    container, user, pass = kb.connection
    @info "KB: Querying with $(length(IOBuffer(cypher_cmd).data)) bytes KB in container $container..."
    @debug "KB: Query:\n$cypher_cmd"
    # Build and run command
    full_cmd = `docker exec --interactive $container cypher-shell --format plain -u $user -p $pass $cypher_cmd`
    if !output
        run(full_cmd, wait=wait)
        return nothing
    else
        result = read(full_cmd, String)
        return parse_neo4j_result(result)
    end
end
