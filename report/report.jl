### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 31594c40-8e7f-11ed-165f-4f09d09a3ec4
begin
	using Markdown
	using PlutoUI
end

# ╔═╡ 899449dc-795c-4ef7-b066-3e69f5d2791b
md"""
# Deep Learning - Neural Networks
## Aristotle Universtity Thessaloniki - School of Informatics
### Assignment 3: PCA with Hebbian Learning
#### Antoniou, Antonios - 9482
#### aantonii@ece.auth.gr
[GitHub repository can be found here](https://github.com/anthonyisafk/deep-learning-hebbian)
"""

# ╔═╡ 112d585f-41b6-4e3b-9881-5436d13da758
md"""
## Introduction

For the final assignment of the Semester for the Deep Learning course, we are given the option to implement a PCA model using Hebbian Learning. The results of that analysis can be then used for less memory and time consuming training of a prediction model, since it will be trained on the transformed dataset, consisting of a **linear combination** of the main components.
\
\
Suppose the number of those components is $C$, and the original number of components (also known as **features**) is $F$. We are then sure that $C\lt F$. This is the reason why the consequent training is executed on less data (but generally speaking no less *information*), therefore in shorter time.
\
\
As usual, before we dive into the specifics of the implementation, we will need to clear up the most significant parts of the terminology.
\
\

### PCA

The idea behind [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) (**Principal Component Analysis**) is that not all features in a dataset have the same gravity, or "explain" the same amount of information. This information, that we also mentioned above, generally has to do with the amount of variance a specific feature is responsible for in the entirety of the matrix of samples.
\
\
This analysis is based on the concept of **eigenvalue decomposition**. A matrix is analysed into its [eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors). The eigenvectors that correspond to the largest eigenvalues are the components that carry the largest percentage of information, which is measured by the variance of the result of the analysis. Ideally, we want to keep about **90% of the original variance**, whenever possible.

### Hebbian Learning

The Hebbian rule is pretty simple and intuitive: The more two neurons agree on an output, the greater the weight between them should be. It can be expanded to the opposite as well: When two neurons output values of different sign, the connection between them should be weakened.
\
\
Mathematically speaking, for two neurons $i$ and $j$, this can be expressed as:
* (1) $\Delta w_{ji}(n)=\eta\cdot y_{j}(n)\cdot x_{i}(n)$,
for a specific moment in time $n$, and a selected learning rate $\eta$. The problem arising from this rule is the lack of boundaries imposed to the weights. Below, we are going to lay out the way in which this challenged is tackled. Now that basics of Hebbian learning are cleared up, what remains to be specified is how PCA can be performed utilizing (1).

### PCA with Hebbian Learning

For an [ANN](https://en.wikipedia.org/wiki/Artificial_neural_network) to perform PCA using the Hebbian rule, there needs to be a much stricter architecture than the usual Networks. More specifically, a Network is able to perform PCA, if it consists of **two layers**: The input layer, and the output layer, with as many nodes as the desired components. This means that the training boils down to training the weights connecting all input nodes with all output nodes.
\
\
The rule in **(1)** is the basis of the training, and to make sure that the solution doesn't diverge we use **Oja's rule**:
* (2) $\Delta w_{ji}(n)=\eta\cdot y_{j}(n)\cdot[x_{i}(n)-y_{j}(n)w_{ji}(n)]$
This way, we can ensure that $|w|=1$.
\
\
However, this algorithm can only work properly when we need to filter out the *principal component* of the features (i.e. when the population of the output layer is 1). In any other case, where $C\gt 1$, the algorithm needs to be enhanced in order to guarantee both convergence and the results we expect. In practice, for the weight between output node $j$ and input node $i$, the update rule will be transformed to:
* (3) $\Delta w_{ji}(n)=\eta\cdot y_{j}(n)\cdot[x_{i}(n)-\sum_{k=1}^{j}y_{k}(n)w_{ki}(n)]$
To get a better grasp of (3): The output neuron with index $j$ represents the j-th component. So for an output neuron $j+1$, to only represent one component, and not accumulate the components before it, the outputs and respective weights have to be subtracted. In this way, the neuron only processes the remainder of the information that was omitted by the previous nodes.

## Implementation

The PCA model was built from scratch based on two classes

### Layer

`Layer` contains all the nodes of a layer, their respective outputs and current weights, together with the arrays keeping the $\Delta w$ values, as well as the $d$ values. The latter is an array of size $(C+1)\times F$. Each row contains $F$ sums (one for every input neuron), that is the result of the sum component in (3). 

### Model

This class encapsulates the two layers of the predictor, and is responsible to fit around the input dataset `X`. Apart from the layers, and corresponding nodes, it contains information on the learning rate $\eta$, the total time of training, and the results of the dimensionality reduction. Last, but certainly not least, it contains the threshold `tolerance`. In short, during each epoch, we keep a metric of the amount of change each weight was subject to, `mean_dw`. Essentially, it is an average of the total changes made to each weight, in proportion to $\eta$ and the number of samples $S$. **Empirically**, it was found that a tolerance of $10^{-9}$ per output node was a low enough value to assume equilibium.
\
\
So, a single iteration in an epoch during training is comprised of simple steps:
* The outputs `y` are calculated using the product of weights and inputs `x`,
* Based on both of the above, we calculate the `d` array, in order to then calculate the array of $\Delta w$ values,
* We calculate `mean_dw` and see if the stability criterion is met. We require:
    * $mean\_dw\leq (C+1)\cdot 10^{-9}$
    * $\#epochs\gt C+1$
Now, why do we use $C+1$ output neurons instead of $C$? It is a solution that was introduced to the "ghost component" that appears in the place of the first principal component (no idea why this is happening). So, when we need 3 components, we train a model for 4, and start from the second one. This means that the calculation of the results of the PCA looks like:
```python
def pca(self, x):
	nsamples = len(x)
	for s in range(nsamples):
		self._comps[s, :] = np.dot(self.layers[1].w[1:,:], x[s])
```

## Testing

The [Body Signals of Smoking](https://www.kaggle.com/datasets/kukuroo3/body-signal-of-smoking) dataset was used. We first conduct some test with PCA from the `sklearn.decomposition` module. The data is **immediately normalized**, so we can avoid overflows during the training of the Hebbian Network. The reason why we also use the normalized data for the direct solution is because we would like to measure the discrepancies between the models with as much common ground as possible.
\
\
We test for number of components $C$, in the range $[1,10]$. This is more than enough testing, since we reach a ratio of $98.602%$ explained variance:

$(LocalResource(
	"../image/sklearn/sklearn_times.png",
	:style =>  "text-align:left;
				width: 45%;"
))
$(LocalResource(
	"../image/sklearn/sklearn_vars.png",
	:style =>  "text-align: right;
				width: 45%;"
))
\
\
The dataset has a column named `ID`. It has no meaningful value, and should be totally discarded by the analysis. For one component, we take normalized data again, but don't drop the column. The result:
```
 ** Total percentage of variance explained: 90.579
```
This is a clear indicator that **we need to study the data we are dealing with**. While sophisticated algorithms can help with the results of any analysis, the first tool we have to use is the knowledge of the dataset, so that we know the measures we will have to take.
"""

# ╔═╡ 833b623e-43c1-4c5d-9dd8-764654086541
md"""
When it comes to the implemented model, once again we test for $C=[1,10]$. We also test for various values of $\eta$, to see if the same results can be achieved through a lesser number of epochs. All tests for $\eta=0.001$ reached the maximum number of epochs allowed by the model, which is **150**. So we narrow down the testing range to $\eta=\{0.005,0.01\}$. The results for $\eta=0.001$ and $\eta=0.005$ are almost identical, so we don't worry about their integrity.
\
\
First we see the number of epochs it took the model to converge, then, for $\eta=0.01$, we take a look at the execution time per epoch, so we can have a measure of the added complexity of an extra output node, and the weight adjustments that come with it.

$(LocalResource(
	"../image/hebbian/strict_hebbian_times.png",
	:style =>  "text-align:left;
				width: 45%;"
))
$(LocalResource(
	"../image/hebbian/strict_hebbian_vars.png",
	:style =>  "text-align: right;
				width: 45%;"
))

| Components | time/epoch | Components | time/epoch |
|------------|------------|------------|------------|
| 1          | 1,808      | 6          | 4,972      |
| 2          | 2,305      | 7          | 6,917      |
| 3          | 2,856      | 8          | 10,423     |
| 4          | 3,895      | 9          | 7,155      |
| 5          | 4,026      | 10         | 7,621      |

As expected, as complexity increases, the time per epoch also does, almost linearly.
"""

# ╔═╡ 1ee44a68-3047-45c8-8c3c-1be08b100b14
md"""
For $C=\{0.05,0.1\}$, we make the equilibrium criterion more lenient (tolerance per output node equal to $10^{-7}$, from $10^{-9}$) and compare the results:

$(LocalResource(
	"../image/hebbian/loose_hebbian_times.png",
	:style =>  "text-align:left;
				width: 45%;"
))
$(LocalResource(
	"../image/hebbian/loose_hebbian_vars.png",
	:style =>  "text-align: right;
				width: 45%;"
))

## Conclusion

For the dataset in question, we achieve comparable -if not identical- results, but at the expense of fitting time. While the standard PCA module consistently takes less than a second to fit, the Hebbian model needs orders of magnitude of that time. The more lenient criterion may cut that time down drastically, but the difference is still vast.

$(LocalResource(
	"../image/comparison.png",
	:style =>  "display: block;
				margin-right: auto;
				margin-left: auto;
				width: 55%;"
))
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.49"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "8eaf9f1b4921132a4cff3f36a1d9ba923b14a481"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.1.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "c47c5fa4c5308f27ccaac35504858d8914e102f9"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.4"

[[IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "f7be53659ab06ddc986428d3a9dcc95f6fa6705a"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "3c837543ddb02250ef42f4738347454f95079d4e"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.3"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "6466e524967496866901a78fca3f2e9ea445a559"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.2"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eadad7b14cf046de6eb41f13c9275e5aa2711ab6"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.49"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[SnoopPrecompile]]
git-tree-sha1 = "f604441450a3c0569830946e5b33b78c928e1a85"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.1"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[Tricks]]
git-tree-sha1 = "6bac775f2d42a611cdfcd1fb217ee719630c4175"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.6"

[[URIs]]
git-tree-sha1 = "ac00576f90d8a259f2c9d823e91d1de3fd44d348"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.4.1"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─31594c40-8e7f-11ed-165f-4f09d09a3ec4
# ╟─899449dc-795c-4ef7-b066-3e69f5d2791b
# ╟─112d585f-41b6-4e3b-9881-5436d13da758
# ╟─833b623e-43c1-4c5d-9dd8-764654086541
# ╟─1ee44a68-3047-45c8-8c3c-1be08b100b14
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
