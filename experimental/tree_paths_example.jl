using Logging
global_logger(ConsoleLogger(stdout, Logging.Info))
using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), ".."))
using Kdautoml
using AbstractTrees
import AbstractTrees: children, printnode

struct Node
    name::String
    children::Vector{Node}
end

function children(n::Node)
    isempty(n.children) && return ()
    return n.children
end

printnode(io::IO, n::Node) = print(io, "•-[$(n.name)]")

node_a = Node("a", Node[])
node_b = Node("b", Node[])
node_c = Node("c", [node_a,node_b])
node_g = Node("g", Node[])
node_d = Node("d", [node_c, node_g])
node_f = Node("f", [])
node_e = Node("e", [node_f])
root = Node("root", [node_d, node_e])
#tree= Tree(root)

print_tree(root)
paths=Kdautoml.extract_paths(root, ["g", "f","a"])
print(paths)
