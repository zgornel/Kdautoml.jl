using Logging
global_logger(ConsoleLogger(stdout, Logging.Debug))
using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), ".."))
using Kdautoml

#kbpath = joinpath(dirname(@__FILE__), "../../../data/knowledge/pipe_synthesis.toml")
@assert length(ARGS) == 1 "A path to the pipeline synthesis KB needs to be provided."
kbpath = ARGS[1]

@info "Loading KB at $kbpath"
kb = Kdautoml.kb_load(kbpath; kb_type=:neo4j, kb_flavour=:pipe_synthesis)

# Load kb data into neo4j db
@info "Loading KB into NEO4J (container=$(kb.connection[1]))"
Kdautoml.KnowledgeBase.execute_kb_query(kb, "MATCH (n) DETACH DELETE n");
for stmt in Kdautoml.KnowledgeBase.kb_to_neo4j_statements(kb)
   Kdautoml.KnowledgeBase.execute_kb_query(kb, stmt)
end
