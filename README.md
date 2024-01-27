[![codecov](https://codecov.io/gh/dynarejulia/FastLapackInterface.jl/branch/main/graph/badge.svg?token=3VH7VTUQNR)](https://codecov.io/gh/dynarejulia/FastLapackInterface.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://dynarejulia.github.io/FastLapackInterface.jl/dev/)

``FastLapackInterface`` separates workspace allocation and actual
running for some Lapack algorithms:
 
 - LU factorization
 - QR factorization
 - Schur factorization
 - Eigen decomposition
 - Bunch-Kaufman decomposition
 - Cholesky decomposition
 - LSE decomposition
 
The API follows the Base julia `LAPACK` definitions and should thus be very transparent.
There is also a unified interface which returns the right `Workspace` to be used with a given
`LAPACK` function.
e.g.
```julia
A = [1.2 2.3
     6.2 3.3]
ws = Workspace(LAPACK.getrf!, A)
decompose!(ws, A)
```
A similar API exists for the above decompositions. For more information and examples please see the [documentation](https://dynarejulia.github.io/FastLapackInterface.jl/dev/).

## Compatibility with MKL
- this package works also with MKL
- OpenBlas and MKL don't compute the required workspace size of some
  functions the same way. 

## Package version
-   v1.x: works only with Julia >= 1.6.3
