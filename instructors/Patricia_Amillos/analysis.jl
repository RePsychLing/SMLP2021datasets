using Arrow, DataFrameMacros, DataFrames, MixedModels
    # read in the data
axb_data = DataFrame(Arrow.Table("./instructors/Patricia_Amillos/data/axb_data.arrow"))

    # for a binary response it can help to summarize the number of true/false responses
acctbl = combine(groupby(axb_data, [:ID, :accuracy]), nrow)
unstack(acctbl, :ID, :accuracy, :nrow)

    # similarly for item
unstack(combine(groupby(axb_data, [:item, :accuracy]), nrow), :item, :accuracy, :nrow)

    # set the contrasts
contr = Dict(
    :ID => Grouping(),
    :item => Grouping(),
    :Task => EffectsCoding(),
    :training => EffectsCoding(),
)

    # fit an initial model with scalar random effects to the hindi contrast group
m1 = fit(
    MixedModel,
    @formula(accuracy ~ 1 + Task * trained + (1|ID) + (1|item)),
    @subset(axb_data, :contrast == "hindi"),
    Bernoulli();
    contrasts=contr,
)

