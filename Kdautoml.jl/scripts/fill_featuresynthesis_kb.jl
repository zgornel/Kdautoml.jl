using Logging
global_logger(ConsoleLogger(stdout, Logging.Debug))
using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), ".."))
using Kdautoml

#kbpath = joinpath(dirname(@__FILE__), "../../../data/knowledge/feature_synthesis.toml")
@assert length(ARGS) == 1 "A path to the feature synthesis KB needs to be provided."
kbpath = ARGS[1]

@info "Loading KB at $kbpath"
kb = Kdautoml.kb_load(kbpath)

# Load kb data into neo4j db
user="neo4j"
pass="test"
container = Kdautoml.get_container(Kdautoml.FEATURESYNTHESIS_CONTAINER_NAME)
@info "Loading KB into NEO4J (container=$container)"
Kdautoml.cypher_shell(container, user, pass, "MATCH (n) DETACH DELETE n");
for stmt in Kdautoml.kb_to_neo4j_statements(kb)
   Kdautoml.cypher_shell(container, user, pass, stmt)
end

