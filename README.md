WORK IN PROGRESS

``FastLapackInterface`` separates workspace allocation and actual
running for some Lapack algorithms:
 
 - LU factorization
 - QR factorization
 - Schur factorization

### LU factorization and linear problem solution

 - ``ws = LinSolveWs(n)`` creates storage for a linear problem with `n`
   equations.
 - ``lu!(A, ws)`` computes the LU factorization and stores it in
   ``ws``
 - ``linsolve_core!(A, b, ws)`` computes LU factorization for matrix
   `A` and returns the solution of `Ax=b` in `b`.
 - ``linsolve_core_no_lu!(A, b, ws)`` computes the solution of `Ax=b`
   and returns it in `b` by using the LU decomposition of `A` already
   stored in `ws`.

### QR factorization ( A = Q*R)

- `ws = QrWs(A)` allocates workspace for QR factorization of
  a matrix similar to `A`
- `geqrf_core!(A, ws)` computes QR factorization of matrix `A` and
  stores it in `A` and `ws.tau` 
- `ormqr_core!(side, A, C, ws)` computes `Q*C` (`side='L'`) or `C*Q`
  (`side='R'`) 
- `ormqr_core!(side, transpose(A), C, ws)` computes `transpose(Q)*C`
  (`side='L'`) or `C*transpose(Q)` (`side='R'`)
  
### Schur factorization

- `DgeesWs(n)` allocates workspace for the real Schur decomposition of
  a matrix of order `n`
- `DgeesWs(A)` allocates workspace for the real Schur decomposition of
  a matrix similar to `A`
- `dgees!(ws, A)` computes the Schur decomposition and stores it in
  `A`. The Schur vectors are stored in `ws.vs` and the eigenvalues in
  `ws.eigen_values`. The Schur decomposition is ordered so that
  eigenvalues larger than 1+1e-6 in modulus are ordered first.
- `dgees!(ws, A, op, crit)` computes the Schur decomposition and stores it in
  `A`. The Schur vectors are stored in `ws.vs` and the eigenvalues in
    `ws.eigen_values`. The Schur decomposition is ordered according to  `λ op crit`
    where `op` can be `<`, `<=`, `>` or `>=`.
- `DggesWs(A, B)` allocates workspace for the real generalized Schur decomposition of
  a matrices similar to `A` and `B`
- `dgges!(jobvsl, jobvsr, A, B, vsl, vsr, eigval, ws)`
   computes the generalized Schur decomposition and stores it in
    `A` and `B`. When `jobvsl` is 'V', the left Schur vectors are stored in `vsl`.
    When `jobvsr` is 'V', the right Schur vectors are stored in `vsr`.
    The eigenvalues are stored in `eigval`. The Schur decomposition is ordered so that
  eigenvalues larger than 1+1e-6 in modulus are ordered first.

## Package version
-   v0.1.3: works only with Julia >= 1.7

## TODO
    - homogenize API to the various functions
