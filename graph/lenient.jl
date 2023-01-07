using DataFrames, CSV
using Plots

filename = "results/smoking_hebbian.csv";
df = DataFrame(CSV.File(filename, header=1));
nrows, ncols = size(df);
df = df[21:40,1:ncols];

eta = [0.05 0.1]

ptimes = plot(
    df[1:10, :nc], [df[1:10, :epochs], df[11:20, :epochs]],
    title = "Loose Hebbian number of epochs",
    xlabel = "Number of components",
    ylabel = "Epochs",
    label = eta,
    legend = :topleft
);
pvars = plot(
    df[1:10, :nc], [df[1:10, :var], df[11:20, :var]],
    title = "Loose Hebbian explained variance",
    xlabel = "Number of components",
    ylabel = "Ratio [%]",
    label = eta,
    legend = :topleft
);

savefig(ptimes, "image/hebbian/loose_hebbian_times.png")
savefig(pvars, "image/hebbian/loose_hebbian_vars.png")