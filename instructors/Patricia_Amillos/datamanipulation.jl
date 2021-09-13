using Arrow, Chain, CSV, DataFrameMacros, DataFrames
      # view this dataframe in table viewer after reading
axb_data = CSV.read("./Patricia_Amillos/full_axb.csv", DataFrame; missingstrings="NA");

      # a brief summary to check on datatypes in columns
describe(axb_data)

      # check for columns that are constant - drop them and Column1, which is row numbers
constants = names(axb_data, nm -> isone(length(unique(axb_data[!, nm]))))

      # reduce the dataframe and transform columns using DataFrameMacros
axb_reduced = @chain axb_data begin
            # drop the constant columns and Column1, which is just the row numbers
    select(Not(push!(constants, "Column1")))
    @transform!(
        :accuracy = :response == :answer_key, # evaluate accuracy as Boolean
        :item = string('I', lpad(:item, 2, '0')),
        :trained = :trained == 1,              # convert from number to Boolean
        :x_dental = :x_dental == 1,
    )
end

      # write the transformed data as an Arrow file using Zstd compression
arrowfile = Arrow.write(
    "./instructors/Patricia_Amillos/data/axb_data.arrow",
    select(   # re-order the columns so the important ones are first
        axb_reduced,
        :ID,
        :item,
        :accuracy,
        :Task,
        :trained,
        :);
    compress = :zstd,
    )
      # check how large the Arrow file is
filesize(arrowfile)

      # as a check, read in the Arrow file and check the column types
axb_arrow = DataFrame(Arrow.Table(arrowfile))
describe(axb_arrow)

