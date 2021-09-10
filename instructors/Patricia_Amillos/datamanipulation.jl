using Arrow, Chain, CSV, DataFrameMacros, DataFrames

axb_data = CSV.read("../../Patricia_Amillos/full_axb.csv", DataFrame; missingstrings="NA"); # view in table viewer

      # check for columns that are constant - drop them and Column1, which is row numbers
constants = names(axb_data, nm -> isone(length(unique(axb_data[:, nm]))))
axb_reduced = @chain axb_data begin
    select(Not(push!(constants, "Column1")))
    @transform!(
        :accuracy = :response == :answer_key, # evaluate accuracy as Boolean
        :item = string('I', lpad(:item, 2, '0')),
        :trained = :trained == 1,              # convert from number to Boolean
        :x_dental = :x_dental == 1,
    )
end
Arrow.write(
    "./data/axb_data.arrow",
    select(   # re-order the columns so the important ones are first
        axb_reduced,
        :ID,
        :item,
        :accuracy,
        :Task,
        :trained,
        :);
    compress = :zstd
    )