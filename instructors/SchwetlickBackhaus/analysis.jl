### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ 6c9f808e-3de8-49af-a45d-56188effc455
begin
	using Arrow
	using DataFrameMacros
	using DataFrames
	using MixedModels
	using PyCall
	using RData
end

# ╔═╡ 250cce64-e730-11eb-347c-0b1bcf5e7d07
md"""
# Data from Daniel Backhaus and Lisa Schwetlick
"""

# ╔═╡ 2391231e-4751-430d-8ccf-707317fff8a6
md"""
Load the data from the .RData file into Julia.
"""

# ╔═╡ 63fc6e8d-c53a-4986-8d7b-f8373ebed015
dat = load("../../SchwetlickBackhaus/smlp_mem_data.RData");

# ╔═╡ 915429f2-0208-4267-b8cc-9ab602b404ff
keys(dat)   # name of the data frame in the .RData file

# ╔═╡ dacdbc08-7805-43e7-910b-4068f05177b9
d = dat["d"]

# ╔═╡ d392c1d3-247f-4de4-b930-92b7b6310b75
describe(d)

# ╔═╡ 39b6e0df-cb98-4c72-9df0-643382c74d8a
md"""
Unfortunately, at present the [`Pluto` package](https://github.com/fonsp/Pluto.jl) and the [`CategoricalArrays` package](https://github.com/JuliaData/CategoricalArrays.jl) don't play nicely together.
To clean this up, we will "round-trip" the data frame to an [`Arrow`](https://arrow.apache.org) representation and back to a DataFrame, first changing the `fixdur` and `trial` columns to integers, which are more compact to store and display,
and converting the `FSP` column from a factor with levels `TRUE` and `FALSE` to a Boolean.

Julius Krumbiegel's `DataFrameMacros` package provides clean ways of doing such transformations.
"""

# ╔═╡ 7899c800-6961-49f7-bd5e-3052edeed70e
filesize(
	Arrow.write(
		"./dat.arrow",
		@transform(
			dat["d"],
			:fixdur = Int16(:fixdur),
			:trial = Int16(:trial),
			:FSP = :FSP == "TRUE",
		);
		compress=:zstd,
	),
)	

# ╔═╡ 36b94384-463e-4a63-aae7-daf9c6ecf679
d1 = DataFrame(Arrow.Table("./dat.arrow"))

# ╔═╡ 7e654ada-6de4-4d8a-bc11-f8bcc249c98f
describe(d1)

# ╔═╡ 802a2c30-91a7-4f2e-9606-cedef9c4daab
md"""
One advantage of using the Arrow storage is that the same file can be accessed by programs in many different languages, including Python and R.
I find that the concise table description from the `pyarrow.feather.read_table` method to be useful in checking the format of the data.
"""

# ╔═╡ 050b2b51-9fb5-406e-9904-563b7d4d5223
feather = pyimport("pyarrow.feather") 

# ╔═╡ 56ad3246-c12c-46a7-9f1f-d689738f21ef
feather.read_table("./dat.arrow")

# ╔═╡ 1004962a-3599-4351-acad-cfab1efb886b
md"""
## Column descriptions

From the description by Daniel and Lisa

* Experiment = we conducted three separate experiments with different subjects. We used same images in all three experiments.The only difference is in the factor similar possible(FSP).
* fixdur = fixation duration our dependent variable
* Img = image number
* trial = trial number
* Subject = subject number
* nthpres = ordinal presentation number of an image: "1", "2"
* imtype = imagetype, is an image: "new", "identical to a former new image" or "similar to a former new image" - this last type we exclude for this analysis
* FPT = factor presentation type: "new", "identical" 
* FSP = factor (now Boolean) similar possible: did the subjects think that similar images could appear? "FALSE" "TRUE

Note: the factor similar possible (FSP) is a between subject factor for experiment 3 and 4 
      BUT a within-subject factor in experiment 5
"""

# ╔═╡ 0f5b7cfb-1c61-4b47-8a49-25f200425d4d
m1 = fit(
	MixedModel,
	@formula(fixdur ~ 1 + FPT + FSP + (1|Subject) + (1|Img)),
	filter(:imtype => !isequal("similar"), d1),
)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Arrow = "69666777-d1a9-59fb-9406-91d4454c9d45"
DataFrameMacros = "75880514-38bc-4a95-a458-c2aea5a3a702"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
MixedModels = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
RData = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"

[compat]
Arrow = "~1.6.0"
DataFrameMacros = "~0.1.0"
DataFrames = "~1.2.0"
MixedModels = "~4.0.0"
PyCall = "~1.92.3"
RData = "~0.8.3"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.6.2"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Arrow]]
deps = ["ArrowTypes", "BitIntegers", "CodecLz4", "CodecZstd", "DataAPI", "Dates", "Mmap", "PooledArrays", "SentinelArrays", "Tables", "TimeZones", "UUIDs"]
git-tree-sha1 = "ca0abf8290cf219aa96a5c89c8210bbf498fecf2"
uuid = "69666777-d1a9-59fb-9406-91d4454c9d45"
version = "1.6.0"

[[deps.ArrowTypes]]
deps = ["UUIDs"]
git-tree-sha1 = "88f8d727e612e3cd5b91c6906b10f25e7821d8d4"
uuid = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
version = "1.1.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "c31ebabde28d102b602bada60ce8922c266d205b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.1"

[[deps.BitIntegers]]
deps = ["Random"]
git-tree-sha1 = "f0853a9d2cf2cf4f9fbd98c6e19ff1627e65a9a7"
uuid = "c3b6d118-76ef-56ca-8cc7-ebb389d030a1"
version = "0.2.4"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "19a35467a82e236ff51bc17a3a44b69ef35185a2"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+0"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "JSON", "Missings", "Printf", "RecipesBase", "Statistics", "StructTypes", "Unicode"]
git-tree-sha1 = "1562002780515d2573a4fb0c3715e4e57481075e"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "0b0aa9d61456940511416b59a0e902c57b154956"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.12"

[[deps.CodecBzip2]]
deps = ["Bzip2_jll", "Libdl", "TranscodingStreams"]
git-tree-sha1 = "2e62a725210ce3c3c2e1a3080190e7ca491f18d7"
uuid = "523fee87-0ab8-5b00-afb7-3ecf72e48cfd"
version = "0.7.2"

[[deps.CodecLz4]]
deps = ["Lz4_jll", "TranscodingStreams"]
git-tree-sha1 = "59fe0cb37784288d6b9f1baebddbf75457395d40"
uuid = "5ba52731-8f18-5e0d-9241-30f10d1ec561"
version = "0.4.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "ded953804d019afa9a3f98981d99b33e3db7b6da"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.0"

[[deps.CodecZstd]]
deps = ["TranscodingStreams", "Zstd_jll"]
git-tree-sha1 = "d19cd9ae79ef31774151637492291d75194fc5fa"
uuid = "6b39b394-51ab-5f42-8807-6242bab2b4c2"
version = "0.7.0"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.Conda]]
deps = ["JSON", "VersionParsing"]
git-tree-sha1 = "299304989a5e6473d985212c28928899c74e9421"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.5.2"

[[deps.Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[deps.DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[deps.DataFrameMacros]]
deps = ["DataFrames"]
git-tree-sha1 = "508d57ef7b78551cf69c2837d80af5017ce57217"
uuid = "75880514-38bc-4a95-a458-c2aea5a3a702"
version = "0.1.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "1dadfca11c0e08e03ab15b63aaeda55266754bad"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.2.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "256d8e6188f3f1ebfa1a5d17e072a0efafa8c5bf"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.10.1"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays"]
git-tree-sha1 = "25b9cc23ba3303de0ad2eac03f840de9104c9253"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "f564ce4af5e79bb88ff1f4488e64363487674278"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.5.1"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "c6a1fff2fd4b1da29d3dccaffb1e1001244d844e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.12"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
deps = ["Test"]
git-tree-sha1 = "15732c475062348b0165684ffe28e85ea8396afc"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.0.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "StructTypes", "UUIDs"]
git-tree-sha1 = "e13a411f59a3f9ad25378b25a40ea08c2ded02e7"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.9.0"

[[deps.JSONSchema]]
deps = ["HTTP", "JSON", "ZipFile"]
git-tree-sha1 = "b84ab8139afde82c7c65ba2b792fe12e01dd7307"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.3"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Lz4_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5d494bc6e85c4c9b626ee0cab05daa4085486ab1"
uuid = "5ced341a-0733-55b8-9ab6-a4889d929147"
version = "1.9.3+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathOptInterface]]
deps = ["BenchmarkTools", "CodecBzip2", "CodecZlib", "JSON", "JSONSchema", "LinearAlgebra", "MutableArithmetics", "OrderedCollections", "SparseArrays", "Test", "Unicode"]
git-tree-sha1 = "575644e3c05b258250bb599e57cf73bbf1062901"
uuid = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"
version = "0.9.22"

[[deps.MathProgBase]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "9abbe463a1e9fc507f12a69e7f29346c2cdc472c"
uuid = "fdba3010-5040-5b88-9595-932c9decdf73"
version = "0.7.8"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[deps.MixedModels]]
deps = ["Arrow", "DataAPI", "Distributions", "GLM", "JSON3", "LazyArtifacts", "LinearAlgebra", "Markdown", "NLopt", "PooledArrays", "ProgressMeter", "Random", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "StatsFuns", "StatsModels", "StructTypes", "Tables"]
git-tree-sha1 = "c17a2cdb285cf936920855e48e198e4f25fcef5c"
uuid = "ff71e718-51f3-5ec2-a782-8ffcbfa3c316"
version = "4.0.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mocking]]
deps = ["ExprTools"]
git-tree-sha1 = "916b850daad0d46b8c71f65f719c49957e9513ed"
uuid = "78c3b35d-d492-501b-9361-3d52fe80e533"
version = "0.7.1"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[deps.NLopt]]
deps = ["MathOptInterface", "MathProgBase", "NLopt_jll"]
git-tree-sha1 = "d80cb3327d1aeef0f59eacf225e000f86e4eee0a"
uuid = "76087f3c-5699-56af-9a33-bf431cd00edd"
version = "0.6.3"

[[deps.NLopt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "2b597c46900f5f811bec31f0dcc88b45744a2a09"
uuid = "079eb43e-fd8e-5478-9966-2cf3e3edb778"
version = "2.7.0+0"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "cde4ce9d6f33219465b55162811d8de8139c0414"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "afadeba63d90ff223a6a48d2009434ecee2ec9e8"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.7.1"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "169bb8ea6b1b143c5cf57df6d34d022a7b60c6db"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.92.3"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[deps.RData]]
deps = ["CategoricalArrays", "CodecZlib", "DataFrames", "Dates", "FileIO", "Requires", "TimeZones", "Unicode"]
git-tree-sha1 = "19e47a495dfb7240eb44dc6971d660f7e4244a72"
uuid = "df47a6cb-8c03-5eed-afd8-b6050d6c41da"
version = "0.8.3"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[deps.Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "ffae887d0f0222a19c406a11c3831776d1383e3d"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.3"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.ShiftedArrays]]
git-tree-sha1 = "22395afdcf37d6709a5a0766cc4a5ca52cb85ea0"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "1.0.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a50550fa3164a8c46747e62063b4d774ac1bcf49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.5.1"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1b9a0f17ee0adde9e538227de093467348992397"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.7"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[deps.StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "ShiftedArrays", "SparseArrays", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "dfdf16cc1e531e154c7e62cd42d531e00f8d100e"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.6.23"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "e36adc471280e8b346ea24c5c87ba0571204be7a"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.7.2"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TimeZones]]
deps = ["Dates", "Future", "LazyArtifacts", "Mocking", "Pkg", "Printf", "RecipesBase", "Serialization", "Unicode"]
git-tree-sha1 = "81753f400872e5074768c9a77d4c44e70d409ef0"
uuid = "f269a46b-ccf7-5d73-abea-4c690281aa53"
version = "1.5.6"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "7c53c35547de1c5b9d46a4797cf6d8253807108c"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.5"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.VersionParsing]]
git-tree-sha1 = "80229be1f670524750d905f8fc8148e5a8c4537f"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.0"

[[deps.ZipFile]]
deps = ["Libdl", "Printf", "Zlib_jll"]
git-tree-sha1 = "c3a5637e27e914a7a445b8d0ad063d701931e9f7"
uuid = "a5390f91-8eb1-5f08-bee0-b1d1ffed6cea"
version = "0.9.3"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "cc4bf3fdde8b7e3e9fa0351bdeedba1cf3b7f6e6"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─250cce64-e730-11eb-347c-0b1bcf5e7d07
# ╠═6c9f808e-3de8-49af-a45d-56188effc455
# ╟─2391231e-4751-430d-8ccf-707317fff8a6
# ╠═63fc6e8d-c53a-4986-8d7b-f8373ebed015
# ╠═915429f2-0208-4267-b8cc-9ab602b404ff
# ╠═dacdbc08-7805-43e7-910b-4068f05177b9
# ╠═d392c1d3-247f-4de4-b930-92b7b6310b75
# ╟─39b6e0df-cb98-4c72-9df0-643382c74d8a
# ╠═7899c800-6961-49f7-bd5e-3052edeed70e
# ╠═36b94384-463e-4a63-aae7-daf9c6ecf679
# ╠═7e654ada-6de4-4d8a-bc11-f8bcc249c98f
# ╟─802a2c30-91a7-4f2e-9606-cedef9c4daab
# ╠═050b2b51-9fb5-406e-9904-563b7d4d5223
# ╠═56ad3246-c12c-46a7-9f1f-d689738f21ef
# ╟─1004962a-3599-4351-acad-cfab1efb886b
# ╠═0f5b7cfb-1c61-4b47-8a49-25f200425d4d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
