# [Fast Lapack Interface](@id man-fastlapack)
The goal of `FastLapackInterface` is to eliminate any temporary allocations when using certain [`LAPACK functions`](@ref LAPACK) compared to Base julia. This is achieved by providing some [`Workspaces`](@ref WorkSpaces) that can then be used during the computation of [`LAPACK functions`](@ref LAPACK).
Eliminating most of the allocations not only improves the computation time of the functions, but dramatically improves `GC` impact when performing multithreaded workloads.

!!! note
    For now the target functionality is limited to [`QR`](@ref QR-id), [`Schur`](@ref Schur-id), [`LU`](@ref LU-id) and [`Eigen`](@ref Eigen-id) related decompositions.

```@meta
DocTestSetup = quote
    using LinearAlgebra, FastLapackInterface
    using LinearAlgebra: LAPACK
end
```

