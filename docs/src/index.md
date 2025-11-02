# [Fast Lapack Interface](@id man-fastlapack)
The goal of `FastLapackInterface` is to eliminate any temporary allocations when using certain [`LAPACK functions`](@ref LAPACK) compared to Base julia. This is achieved by providing some [`Workspaces`](@ref WorkSpaces) that can then be used during the computation of [`LAPACK functions`](@ref LAPACK).
Eliminating most of the allocations not only improves the computation time of the functions, but dramatically improves `GC` impact when performing multithreaded workloads.

```@meta
DocTestSetup = quote
    using LinearAlgebra, FastLapackInterface
    using LinearAlgebra: LAPACK
DocTestFilters = r"(\d*)\.(\d{15})\d+" => s"\1.\2"
end
```

