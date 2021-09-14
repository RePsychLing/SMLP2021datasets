using Arrow, CSV

csvs = filter(readdir(".")) do f
    return last(splitext(f)) == ".csv"
end

for f in csvs    
    Arrow.write("$(first(splitext(f))).arrow", CSV.File(f); compress=:zstd)
end

