# Julia v1.7.0 speed versus Julia v1.6.2

Kyla McConnell provides model specifications in [a repository](https://github.com/kyla-mcconnell/inddiff_experience).
At present the data have not been released to the public but Kyla was kind enough to provide me with a copy for illustration here.

## Use of optimized BLAS

The parameter estimates are optimized by taking some parameter values, doing some linear algebra, evaluating the objective, updating the parameter estimates and repeating until convergence.
In these steps almost all the time time is spent in the linear algebra computations.

Because numerical linear algebra often constitutes a substantial portion of the execution time, it has been highly optimized in what are called the Basic Linear Algebra Subroutines (BLAS).
By default Julia uses [OpenBLAS](https://www.openblas.net).
Intel provides an even more optimized version [MKL](https://en.wikipedia.org/wiki/Math_Kernel_Library), which, unsurprisingly, works best on Intel processors.
MKL is not open source but is freely available now and Julia has capabilities of replacing OpenBLAS with MKL.

The process of replacing OpenBLAS by MKL was complicated.
Starting with Julia v1.7.0 the process has been simplified considerably.

## Fitting a model to Kyla's data in v1.6.2

Because timings are strongly affected by running Zoom at the same time I will just quote the results here.
The latest version of MixedModels when run in the REPL or VS Code presents timing information on the optimization phase only.
```julia
julia> cntrsts = Dict(
           :position => EffectsCoding(base="spillover_2"),
           :education => HelmertCoding(levels=["High school", "Trade school", "Undergraduate", "Grad school"]),
           :id => Grouping(),
           :w1 => Grouping(),
           :w2 => Grouping(),
       );

julia> formula_maximal_ftp = @formula (logRT  ~ 1 + FTP_lz * reading_exp_z * age_z * position + trial_number_z + word_number_z + length_z + prev_length_z + education +
                      (1 + FTP_lz + trial_number_z + word_number_z + length_z + prev_length_z | id) + 
                      (1 + trial_number_z + reading_exp_z * age_z | w1) +
                      (1 + trial_number_z + reading_exp_z * age_z | w2));

julia> formula_maximal_btp = @formula (logRT  ~ 1 + BTP_lz * reading_exp_z * age_z * position + trial_number_z + word_number_z + length_z + prev_length_z +  education + 
                      (1 + BTP_lz + trial_number_z + word_number_z + length_z + prev_length_z | id) + 
                      (1 + trial_number_z + reading_exp_z * age_z | w1) +
                      (1 + trial_number_z + reading_exp_z * age_z | w2));

julia> versioninfo()
Julia Version 1.6.2
Commit 1b93d53fc4 (2021-07-14 15:36 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-11.0.1 (ORCJIT, tigerlake)
Environment:
  JULIA_EDITOR = code
  JULIA_NUM_THREADS = 6

julia> maximal_ftp = fit(MixedModel, formula_maximal_ftp, spr_critregion, contrasts = cntrsts);
Minimizing 2739  Time: 0 Time: 0:02:19 (50.97 ms/it)
  objective:  1278.5333022403308
```

## Using Julia v1.7.0

```julia
julia> maximal_ftp = fit(MixedModel, formula_maximal_ftp, spr_critregion, contrasts = cntrsts);
Minimizing 2739  Time: 0 Time: 0:01:57 (42.78 ms/it)
  objective:  1278.5333022403308

julia> versioninfo()
Julia Version 1.7.0-beta4
Commit d0c90f37ba (2021-08-24 12:35 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, tigerlake)

julia> using MKL

julia> maximal_ftp = fit(MixedModel, formula_maximal_ftp, spr_critregion, contrasts = cntrsts);
Minimizing 2753  Time: 0 Time: 0:00:45 (16.59 ms/it)
  objective:  1278.5332993768975
```
