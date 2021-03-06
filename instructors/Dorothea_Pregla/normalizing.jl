### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ a31887be-d60b-41a8-a561-3ff0f6491e7b
using CSV, DataFrames

# ╔═╡ 1214f8e0-d687-11eb-296c-977d3ccd3c36
md"""
# Normalizing data tables

The popularity of the [tidyverse](https://github.com/tidyverse/tidyverse) collection of packages for R has led to many public data sets being presented as tidy data frames.

## Redundancies and possible inconsistency

As an organizing principle, having a tidy data frame is desirable but it often results in redundancies in the data, as we will see below.
Redundancies are usually not a problem in terms of the amount of storage taken up; storage, even on a public cloud, is very low cost now.
However, the redundancies present the possibility of accidental inconsistencies, which we do want to avoid.

Those working with relational data bases often emphasize the use of a *normalized* database where redundant information is eliminated and the table to be used, e.g. for fitting models, is generated by *joining* several tables of non-redundant information together.

The [DataFrames.jl package](https://github.com/JuliaData/DataFrames.jl) for Julia provides the ability to join tables flexibly and efficiently.  In fact, much of the recent development work on that package, primarily by Bogumił Kamiński, has been in making joins more efficient.

### General approach

The general approach is to isolate those covariates or columns that are characteristics of the subject in a *subject table*.
In terms of the experimental design these covariates are *between-subject* in that they vary between the subjects but not within a subject.
For example, demographic variables are usually between-subject.

Similarly the covariates that are characteristic of the items are *between-item* and are stored in an *item table*.
Word frequency might be an example of a *between-item* covariate.

The *observation table* is a tidy table containing one row for each observation of the response plus the subject identifier, the item identifier and any covariates that vary within-subject and within-item.

When between-subject or between-item covariates are required for exploratory analysis or model fitting the model table is generated from the observation table, the subject table and the item table by a join.

### Data set

First load the packages to be used, read the .csv file, and convert it to a DataFrame.
"""

# ╔═╡ 1ad1a572-22e4-4c9d-b893-95f270c795d0
orig = CSV.read("../../Dorothea_Pregla/SPL_decl.csv", DataFrame; missingstring="NA") 

# ╔═╡ 96410f16-b1a8-4207-802b-abb7eeb1d919
size(orig)

# ╔═╡ 07967240-57aa-42ac-aa4f-c74d83055986
describe(orig)

# ╔═╡ 07cbb2a4-7c37-4a31-8b10-88f478b0b4f8
length(unique(orig.subj))

# ╔═╡ 3fd79669-b8fb-40e7-ba5d-6d9339e01f50
md"""
These data were provided by Dorothea Pregla.
Her description of the columns is:
- subj: subject number 
- item: number of the item (number of items: n=20)
- condition: a (canonical) and b (non-canonical), we did not use a latin square design, i.e., all participants saw both conditions of all items
- region: region in the sentence (0, 1, 2, 3, question)
  - 0: sentence onset ("hier")
  - 1: verb ("tröstet")
  - 2: NP1 ("der Tiger")
  - 3: adverb ("gerade")
  - question: NP2 + picture selection ("den Esel" + picture selection)
- subj_status: healthy control (HC) or individual with aphasia (IWA)
- sentence: the respective auditory presented sentence
- word: the respective word of the sentence
- rt: listening time for the respective region
- acc: picture selection accuracy
- age, sex, years of education: demographic data of the participant
- aphasia_type: Aachen aphasia test syndrome, NA for control participants
- digit_symbol_substitution: WAIS subtest, number of correct items in two minutes
- digit span, block span: WMS-R subtests, forward and backward raw scores and longest span
- picture span: working memory test, forward and backward raw scores and longest span
- lexdec: lexical decision task, mean reaction times in the task

Many of these columns are characteristics of the subject, of whom there are 71.
These characteristics are repeated to fill out the 14185 rows of the corresponding column in the table.  This is the nature of the redundancy.

Furthermore, if there was some kind of an error and, say, a subject's age was recorded inconsistently in one or more rows of this table, this would not be automatically detected in this format.

A way to enforce consistency is to create one table for the subjects with 71 rows and all the subject-specific columns recorded only once in there.

Those who collected the data would know what columns are subject-specific and we could procede to `select` those columns, along with the `subj` column, and apply `unique` to get the subject table.
We should, of course, check that the number of unique rows in the subject table is exactly 71, the number of subjects.

For the sake of illustration, we will show a way of determining which columns are properties of the subject for which we will use a `GroupedDataFrame`.
"""

# ╔═╡ fb224e36-4427-43f9-a435-d8a7c7c880a3
gdf = groupby(orig, :subj)

# ╔═╡ a90307e7-764c-41e1-a77a-7f6fc30c275b
md"""
We are trying to find the columns that are constant for each subject.
For example, the `subj_status` should be constant for each subject but the `condition` is not.

One way of checking if a vector is constant is to compare all the values to the first one.
A simple version of an `isconstant` function is
"""

# ╔═╡ 13fc105c-3e6e-492b-859c-08f4801c4bdb
isconstant1(v) = all(==(first(v)), v)

# ╔═╡ d3e8956c-87fc-406c-9099-ed78a46d98fd
md"""
where `==(first(v))` creates a function of the form x -> x == first(v) but with a bit less overhead.

This works for simple cases
"""

# ╔═╡ 2e43c5ac-ba4b-4214-863b-cc5fa081169b
isconstant1(first(gdf).age)

# ╔═╡ 934fe224-91fb-41f1-808a-62ea9fc63146
isconstant1(first(gdf).item)

# ╔═╡ 0dc9be9d-28af-4062-945a-61e529200647
md"""
but it has (at least) two problems on edge cases.  First, what if `v` is empty?  The call to `first(v)` will fail.
"""

# ╔═╡ c55be9cf-83e8-4d36-b2c8-01763cfd2020
isconstant1([])

# ╔═╡ 07689442-b01f-4727-84c9-921ee669d7db
md"""
It is reasonably straightforward to avoid this situation by checking first if `v` is empty, in which case we will return `true`, arguing that a vector with no elements is constant.
"""

# ╔═╡ 2ebdf2c4-3234-4e1a-a122-10603b77e4ef
function isconstant2(v)
	isempty(v) || all(==(first(v)), v)
end

# ╔═╡ 757ba155-6dc5-49c5-beea-826ec8efa560
isconstant2([])

# ╔═╡ 0021dfba-e466-4ff4-8f3d-f521e47bbbed
md"""
The second edge case involves missing data.  Comparisons with `missing` always return `missing` not `true` or `false`.
"""

# ╔═╡ 652cf724-f1fa-46c9-be97-8b52c71b606c
isconstant2(first(gdf).aphasia_type)

# ╔═╡ 24dcaf32-e907-41ab-a7b9-099f6b329d2f
md"""
If all of the elements of `v` are missing we conclude that `v` is constant otherwise if the comparison returns `missing` then it is not constant.
We write our final version of `isconstant` using `coalesce`, which is a function that returns the first non-missing value in its argument list.
"""

# ╔═╡ d85cbb0d-6671-4ee0-a013-d5927a33b1bb
function isconstant(v)
	return isempty(v) || all(ismissing, v) || coalesce(all(==(first(v)), v), false)
end

# ╔═╡ 56222937-e45e-4e7a-b49b-d5fb9bace1a7
md"Next we create a method that takes a `GroupedDataFrame` and a name `nm` and iterates over all the subdataframes in `gdf`."

# ╔═╡ a237208d-1456-4fc6-95e8-c1bd10d20c70
isconstant(gdf::GroupedDataFrame, nm) = all(sdf -> isconstant(sdf[!, nm]), gdf) 

# ╔═╡ a1c38547-9d80-472c-af21-0e9db3dd41d7
isconstant(gdf, :subj_status)

# ╔═╡ 6bb74de8-d8ea-4674-8419-35a33d8f7533
isconstant(gdf, :item)

# ╔═╡ dbef0ab4-e103-438d-93c2-98d4bd4920ad
isconstant(gdf, :lexdec)

# ╔═╡ 0045e306-8123-4af0-a6e5-70407ef9e6fb
md"""
Finally, we iterate over all the subdataframes in the `GroupedDataFrame` and all the column names, checking this property.
"""

# ╔═╡ be1ec22f-c3f4-447e-8bf4-a301db064b9f
subj_cols = filter(nm -> isconstant(gdf, nm), names(orig))

# ╔═╡ 11d893ed-d4f3-43a6-b0bf-1746336936dd
md"""
Create `subjtable` and check that it has the correct number of rows (71).
"""

# ╔═╡ 242b0b14-4157-4335-b30c-295c2374d5a4
subjtable = unique(select(orig, subj_cols))

# ╔═╡ ef685a16-c24f-4f8b-958d-3f4eae9ec76e
md"""
Next we check if there are columns that are constant across items.
"""

# ╔═╡ 65f93aff-5453-4f90-93d4-d6ad1686abd8
item_cols = let
	gdfitem = groupby(orig, :item)
	filter(nm -> isconstant(gdfitem, nm), names(orig))
end

# ╔═╡ 77a7b199-adf2-426d-8df5-c9327b61aab2
md"""
So `item` is the only column that is constant within items.

Finally we create a response table by removing the columns from `subjtable`, except for `subj` itself.
"""

# ╔═╡ 8e4cadd0-b5f9-42b5-a8b7-bc28551a35e6
responsetable = select(orig, :subj, setdiff(names(orig), subj_cols))

# ╔═╡ 92aacaba-f831-4a8e-a023-d5920de6f0b4
names(responsetable)

# ╔═╡ 1479ed0c-8d07-4fbf-a897-05e5a937127d
md"""
To regenerate the original table (with, perhaps, a different ordering of the columns), we use a `leftjoin` on the `responsetable` and `subjtable` on `:subj`.
"""

# ╔═╡ 6bc71c0f-1c0b-4995-a245-6044a2d275f4
begin
	newtable = leftjoin(responsetable, subjtable, on = :subj)
	describe(newtable)
end

# ╔═╡ 827ac2ca-70e7-4ff1-b414-b8310669fdf4
md"""
## General approach

In most cases of "subject/item" data like this we expect to have 3 tables: a subject table that contains the subject identifier and any columns that are characteristics of the subjects, an item table, and an observation table that contains the subject, the item, any within-subject/within-item experimental conditions and the response.
In this case because the item table is trivial, containing only the item column itself, it can be omitted.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"

[compat]
CSV = "~0.8.5"
DataFrames = "~1.1.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.6.2"
manifest_format = "2.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CSV]]
deps = ["Dates", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "Tables", "Unicode"]
git-tree-sha1 = "b83aa3f513be680454437a0eee21001607e5d983"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.8.5"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[deps.Crayons]]
git-tree-sha1 = "3f71217b538d7aaee0b69ab47d9b7724ca8afa0d"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.0.4"

[[deps.DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Reexport", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "66ee4fe515a9294a8836ef18eea7239c6ac3db5e"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.1.1"

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

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

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

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

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

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "Markdown", "Reexport", "Tables"]
git-tree-sha1 = "0d1245a357cc61c8cd61934c07447aa569ff22e6"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "1.1.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

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

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

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

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─1214f8e0-d687-11eb-296c-977d3ccd3c36
# ╠═a31887be-d60b-41a8-a561-3ff0f6491e7b
# ╠═1ad1a572-22e4-4c9d-b893-95f270c795d0
# ╠═96410f16-b1a8-4207-802b-abb7eeb1d919
# ╠═07967240-57aa-42ac-aa4f-c74d83055986
# ╠═07cbb2a4-7c37-4a31-8b10-88f478b0b4f8
# ╟─3fd79669-b8fb-40e7-ba5d-6d9339e01f50
# ╠═fb224e36-4427-43f9-a435-d8a7c7c880a3
# ╟─a90307e7-764c-41e1-a77a-7f6fc30c275b
# ╠═13fc105c-3e6e-492b-859c-08f4801c4bdb
# ╟─d3e8956c-87fc-406c-9099-ed78a46d98fd
# ╠═2e43c5ac-ba4b-4214-863b-cc5fa081169b
# ╠═934fe224-91fb-41f1-808a-62ea9fc63146
# ╟─0dc9be9d-28af-4062-945a-61e529200647
# ╠═c55be9cf-83e8-4d36-b2c8-01763cfd2020
# ╟─07689442-b01f-4727-84c9-921ee669d7db
# ╠═2ebdf2c4-3234-4e1a-a122-10603b77e4ef
# ╠═757ba155-6dc5-49c5-beea-826ec8efa560
# ╟─0021dfba-e466-4ff4-8f3d-f521e47bbbed
# ╠═652cf724-f1fa-46c9-be97-8b52c71b606c
# ╟─24dcaf32-e907-41ab-a7b9-099f6b329d2f
# ╠═d85cbb0d-6671-4ee0-a013-d5927a33b1bb
# ╟─56222937-e45e-4e7a-b49b-d5fb9bace1a7
# ╠═a237208d-1456-4fc6-95e8-c1bd10d20c70
# ╠═a1c38547-9d80-472c-af21-0e9db3dd41d7
# ╠═6bb74de8-d8ea-4674-8419-35a33d8f7533
# ╠═dbef0ab4-e103-438d-93c2-98d4bd4920ad
# ╟─0045e306-8123-4af0-a6e5-70407ef9e6fb
# ╠═be1ec22f-c3f4-447e-8bf4-a301db064b9f
# ╟─11d893ed-d4f3-43a6-b0bf-1746336936dd
# ╠═242b0b14-4157-4335-b30c-295c2374d5a4
# ╟─ef685a16-c24f-4f8b-958d-3f4eae9ec76e
# ╠═65f93aff-5453-4f90-93d4-d6ad1686abd8
# ╟─77a7b199-adf2-426d-8df5-c9327b61aab2
# ╠═8e4cadd0-b5f9-42b5-a8b7-bc28551a35e6
# ╠═92aacaba-f831-4a8e-a023-d5920de6f0b4
# ╟─1479ed0c-8d07-4fbf-a897-05e5a937127d
# ╠═6bc71c0f-1c0b-4995-a245-6044a2d275f4
# ╟─827ac2ca-70e7-4ff1-b414-b8310669fdf4
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
