# [Fast Lapack Interface](@id man-fastlapack)
The goal of `FastLapackInterface` is to eliminate any temporary allocations when using certain LAPACK functions compared to Base julia.
This is achieved by providing some structures that represent [`Workspaces`](@ref WorkSpaces) that can then be used during the computation of [`LAPACK functions`](@ref LAPACK).

!!! note
    For now the target functionality is limited to [`QR`](@ref QR-id), [`Schur`](@ref Schur-id) and [`LU`](@ref LU-id) related decompositions.

```@meta
DocTestSetup = quote
    using LinearAlgebra, FastLapackInterface
    using LinearAlgebra: LAPACK
end
```

