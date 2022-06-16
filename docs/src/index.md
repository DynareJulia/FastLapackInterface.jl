# [Fast Lapack Interface](@id man-fastlapack)
The goal of FastLapackInterface is to eliminate any temporary allocations when using LAPACK functions as defined in Base julia.
This is achieved by providing some structures that represent workspaces that can then be used inside the LAPACK functions.
The target functionality are some of the factorizations and decompositions defined in the LinearAlgebra stdlib.

```@meta
DocTestSetup = :(using LinearAlgebra, FastLapackInterface)
```

```@docs
FastLapackInterface.LUWs
```
