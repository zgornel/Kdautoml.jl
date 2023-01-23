# Run this with 
# $ julia manual_pipe.jl
# ...
#
using DelimitedFiles, StatsBase, LIBSVM, UnicodePlots

const MAX_STMT = 30

function _apply(fstr, args)
    #f = eval(Meta.parse(fstr))
    stmt = "($fstr)($( join(args,",") ))"
    @info "pushing $(stmt[1:min(length(stmt),MAX_STMT)])[...], args=$args"
    eval(Meta.parse(stmt))
end


# Data loading
f1 = "(header, args...)->header ? readdlm(args...; header)[1] : readdlm(args...; header)"
v1 = _apply(f1, ["true", "\"iris.csv\"", "','"]) # add type which is the semantical information

# Pre-processing
f2 = """
     (data, idxs)->begin
        n = size(data,2);
        tr=sample(1:n, round(Int, n*0.6), replace=false);
        ts=setdiff(1:n,tr);
        (data = Matrix{Float64}(getindex(data, :, idxs)|>transpose),
         labels = string.(getindex(data, :, setdiff(1:n, idxs))),
         split=(train=tr, test=ts))
      end
      """
v2 = _apply(f2, ["v1", "[1,2,3,4]"]) # add type which is the semantical information

# Plot data
println(scatterplot(v2.data[1,:], v2.data[2,:], color=:white))

f3 = """
    (input)->begin
        idxs = input.split.train
        trlabels = input.labels[idxs]
        trdata = input.data[:, idxs]
        (input ..., model=svmtrain(trdata, trlabels))
    end
    """

v3 = _apply(f3, ["v2"])


f4 = """
    input->begin
        data, labels, (tridxs,tsidxs), model = input
        @info \"accuracy = \$(mean(labels[tsidxs].== svmpredict(model, data[:,tsidxs])[1]))\";
    end
    """
v4 = _apply(f4, ["v3"])
