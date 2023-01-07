using DataFrames, CSV
using Plots

file_sklearn = "results/smoking_pca.csv";
file_hebbian = "results/smoking_hebbian.csv";
sklearn = DataFrame(CSV.File(file_sklearn, header=1));
hebbian = DataFrame(CSV.File(file_hebbian, header=1));
hebbian = hebbian[31:40, [:nc, :var]]

p = plot(
    hebbian[1:10, :nc], [hebbian[1:10, :var], sklearn[1:10, :var]],
    title = "Comparison between loose Hebbian and sklearn",
    xlabel = "Number of components",
    ylabel = "Ratio [%]",
    label = ["Hebbian" "sklearn"],
    legend = :topleft
);

savefig(p, "image/comparison.png");
