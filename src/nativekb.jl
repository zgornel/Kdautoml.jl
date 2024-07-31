struct KnowledgeBaseNative <: AbstractKnowledgeBase
    data::Dict
end

Base.show(io::IO, kb::KnowledgeBaseNative) = begin
    mb_size = Base.summarysize(kb.data)/(1024^2)
    print(io, "KnowledgeBase (native), $mb_size MB of data")
end


# The methods for the KB interface
function build_ps_query end

function build_fs_query end

function execute_kb_query end


# Actual implementation
function build_ps_query(action,
                        kb::Type{KnowledgeBaseNative};
                        precondition_symbols=DEFAULT_PRECONDITION_SYMBOLS)
    @warn "`build_ps_query` not implemented for kb::$(typeof(kb))"
    return ""
end


function build_fs_query(feature_type, kb::Type{KnowledgeBaseNative})
    @warn "`build_fs_query` not implemented for kb::$(typeof(kb))"
    return ""
end

function execute_kb_query(kb::KnowledgeBaseNative, cypher_cmd; wait=true, output=false)
    @warn "`execute_kb_query` not implemented for kb::$(typeof(kb))"
    return String[;;]
end
