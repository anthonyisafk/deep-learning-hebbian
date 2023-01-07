using DataFrames, CSV
using Plots

filename = "results/smoking_hebbian.csv";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = df[1:20,1:ncols];

eta = [0.005 0.01]

ptimes = plot(
    df[1:10, :nc], [df[1:10, :epochs], df[11:20, :epochs]],
    title = "Strict Hebbian number of epochs",
    xlabel = "Number of components",
    ylabel = "Epochs",
    label = eta,
    legend = :topleft
);
pvars = plot(
    df[1:10, :nc], [df[1:10, :var], df[11:20, :var]],
    title = "Strict Hebbian explained variance",
    xlabel = "Number of components",
    ylabel = "Ratio [%]",
    label = eta,
    legend = :topleft
);

savefig(ptimes, "image/hebbian/strict_hebbian_times.png")
savefig(pvars, "image/hebbian/strict_hebbian_vars.png")
