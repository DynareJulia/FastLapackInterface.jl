[![codecov](https://codecov.io/gh/louisponet/FastLapackInterface.jl/branch/main/graph/badge.svg?token=3VH7VTUQNR)](https://codecov.io/gh/louisponet/FastLapackInterface.jl)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://louisponet.github.io/FastLapackInterface.jl/dev)

``FastLapackInterface`` separates workspace allocation and actual
running for some Lapack algorithms:
 
 - LU factorization
 - QR factorization
 - Schur factorization

### LU factorization and linear problem solution

 - ``ws = LUWs(A)`` creates storage for a linear problem with matrix representation `A`.
 - ``LAPACK.getrf!(A, ws)`` computes the LU factorization using the preallocated workspace and returns the
 arguments that can be used to construct `LinearAlgebra.LU`.
 
```julia
A = [1.2 2.3
     6.2 3.3]
ws = LUWs(A)
LinearAlgebra.LU(LAPACK.getrf!(A, ws)...)
```

### QR factorization ( A = Q*R)

- `ws = QRWs(A)` allocates workspace for QR factorization of
  a matrix similar to `A`
- `geqrf!(A, ws)` computes QR factorization of matrix `A` and
  stores it in `A` and `ws.τ`, and returns the arguments for the constructor of `LinearAlgebra.QR`.  
- `ormqr!(side, A, C, ws)` computes `Q*C` (`side='L'`) or `C*Q`
  (`side='R'`) 
- `ormqr!(side, transpose(A), C, ws)` computes `transpose(Q)*C`
  (`side='L'`) or `C*transpose(Q)` (`side='R'`)
- Workspaces exist for `QRCompactWY` (`QRWYWs`) and `QRPivoted` (`QRpWs`).

```julia
A = [1.2 2.3
     6.2 3.3]
ws = QRWs(A)
LinearAlgebra.QR(LAPACK.geqrf!(A, ws)...)
```
  
### Schur factorization

- `SchurWs(A)` allocates workspace for the real Schur decomposition of
  a matrix similar to `A`.
- `gees!(A, ws)` computes the Schur decomposition and returns the arguments to
   the constructor of `LinearAlgebra.Schur`.
- A Workspace for the generalized schur decomposition also exists.
- It is possible to use select functions with `gees!` or `gges!` to order the eigenvalues.
```julia
A = [1.2 2.3
     6.2 3.3]
ws = SchurWs(A)
LinearAlgebra.Schur(LAPACK.gees!('V', A, ws)...)
```
For more info see the [Documentation](https://louisponet.github.io/FastLapackInterface.jl/dev)

## Package version
-   v0.1.3: works only with Julia >= 1.7
