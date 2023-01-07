using DataFrames, CSV
using Plots

filename = "results/smoking_pca.csv";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df)


ptimes = plot(
    df[!, :nc], df[!, :time],
    title = "sklearn fitting times",
    xlabel = "Number of components",
    ylabel = "Time [sec.]",
    legend = false
)

pvars = plot(
    df[!, :nc], df[!, :var],
    title = "sklearn explained variance ratio",
    xlabel = "Number of components",
    ylabel = "Ratio [%]",
    legend = false
)

savefig(ptimes, "image/sklearn/sklearn_times.png")
savefig(pvars, "image/sklearn/sklearn_vars.png")