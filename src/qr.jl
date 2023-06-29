# TODO: LinearAlgebra.qr! method that works with any workspace
import LinearAlgebra.LAPACK: geqrf!, ormqr!, geqrt!, geqp3!, orgqr!, orgql!

"""
    QRWs

Workspace for standard [`LinearAlgebra.QR`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.QR)
factorization using the [`LAPACK.geqrf!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = QRWs(A)
QRWs{Float64}
  work: 64-element Vector{Float64}
  τ: 2-element Vector{Float64}

julia> t = QR(LAPACK.geqrf!(ws, A)...)
QR{Float64, Matrix{Float64}, Vector{Float64}}
Q factor:
2×2 LinearAlgebra.QRPackedQ{Float64, Matrix{Float64}, Vector{Float64}}:
 -0.190022  -0.98178
 -0.98178    0.190022
R factor:
2×2 Matrix{Float64}:
 -6.31506  -3.67692
  0.0      -1.63102

julia> Matrix(t)
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3
```
"""
struct QRWs{T<:Number} <: Workspace
    work::Vector{T}
    τ::Vector{T}
end

Base.length(ws::QRWs) = length(ws.τ)

for (geqrf, elty) in ((:dgeqrf_, :Float64),
                      (:sgeqrf_, :Float32),
                      (:zgeqrf_, :ComplexF64),
                      (:cgeqrf_, :ComplexF32))
    @eval begin
        function Base.resize!(ws::QRWs, A::StridedMatrix{$elty}; work=true)
            m, n = size(A)
            lda = max(1, stride(A, 2))
            resize!(ws.τ, min(m, n))
            if work
                info = Ref{BlasInt}()
                ccall((@blasfunc($geqrf), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                      m, n, A, lda, ws.τ, ws.work, -1, info)
                chklapackerror(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
        QRWs(A::StridedMatrix{$elty}) =
            resize!(QRWs(Vector{$elty}(undef, 1), Vector{$elty}(undef, 1)), A)

        function geqrf!(ws::QRWs, A::AbstractMatrix{$elty}; resize=true)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            nws = length(ws)
            minn = min(m, n)
            if nws != minn 
                if resize
                    resize!(ws, A; work = minn > nws)
                else
                    throw(WorkspaceSizeError(nws, minn))
                end
            end
            lda = max(1, stride(A, 2))
            lwork = length(ws.work)
            info = Ref{BlasInt}() # This actually doesn't cause allocations
            ccall((@blasfunc($geqrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, ws.τ, ws.work, lwork, info)
            chklapackerror(info[])
            return A, ws.τ
        end
    end
end
"""
    geqrf!(ws, A; resize=true) -> (A, ws.τ)

Compute the `QR` factorization of `A`, `A = QR`, using previously allocated [`QRWs`](@ref) workspace `ws`.
`ws.τ` contains scalars which parameterize the elementary reflectors of the factorization.
`ws.τ` must have length greater than or equal to the smallest dimension of `A`.
If this is not the case, and `resize==true` the workspace will be automatically
resized to the appropriate size.

`A` and `ws.τ` modified in-place.
"""
geqrf!(ws::QRWs, A::AbstractMatrix; kwargs...)


"""
    QRWYWs

Workspace to be used with the [`LinearAlgebra.QRCompactWY`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.QRCompactWY)
representation of the blocked QR factorization which uses the [`LAPACK.geqrt!`](@ref) function.
By default the blocksize for the algorithm is taken as `min(36, min(size(template)))`, this can be
overridden by using the `blocksize` keyword of the constructor.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = QRWYWs(A)
QRWYWs{Float64, Matrix{Float64}}
  work: 4-element Vector{Float64}
  T: 2×2 Matrix{Float64}

julia> t = LinearAlgebra.QRCompactWY(LAPACK.geqrt!(ws, A)...)
LinearAlgebra.QRCompactWY{Float64, Matrix{Float64}, Matrix{Float64}}
Q factor:
2×2 LinearAlgebra.QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}:
 -0.190022  -0.98178
 -0.98178    0.190022
R factor:
2×2 Matrix{Float64}:
 -6.31506  -3.67692
  0.0      -1.63102

julia> Matrix(t)
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3
```
"""
mutable struct QRWYWs{R<:Number,MT<:StridedMatrix{R}} <: Workspace
    work::Vector{R}
    T::MT
end

function Base.resize!(ws::QRWYWs, A::StridedMatrix; blocksize=36, work=true)
    require_one_based_indexing(A)
    chkstride1(A)
    m, n = BlasInt.(size(A))
    @assert n > 0 ArgumentError("Not a Matrix")
    m1 = min(m, n)
    nb = min(m1, blocksize)
    ws.T = similar(ws.T,  nb, m1)
    if work
        resize!(ws.work, nb*n)
    end
    return ws
end

QRWYWs(A::StridedMatrix{T}; kwargs...) where {T <: LinearAlgebra.BlasFloat} =
    resize!(QRWYWs(T[], Matrix{T}(undef, 0, 0)), A; kwargs...)

for (geqrt, elty) in ((:dgeqrt_, :Float64),
                      (:sgeqrt_, :Float32),
                      (:zgeqrt_, :ComplexF64),
                      (:cgeqrt_, :ComplexF32))
    @eval function geqrt!(ws::QRWYWs, A::AbstractMatrix{$elty}; resize=true)
        require_one_based_indexing(A)
        chkstride1(A)
        m, n = size(A)
        minmn = min(m, n)
        nb = size(ws.T, 1)
        if nb != minmn
            if resize
                resize!(ws, A, work = nb < minmn)
            else
                throw(WorkspaceSizeError(nb, minmn))
            end
        end
        nb = size(ws.T, 1)

        lda = max(1, stride(A, 2))
        work = ws.work
        info = Ref{BlasInt}()
        ccall((@blasfunc($geqrt), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                Ptr{BlasInt}),
                m, n, nb, A,
                lda, ws.T, max(1, stride(ws.T, 2)), ws.work,
                info)
        chklapackerror(info[])
        return A, ws.T
    end
end

"""
    geqrt!(ws, A; resize=true) -> (A, ws.T)

Compute the blocked `QR` factorization of `A`, `A = QR`, using a preallocated [`QRWYWs`](@ref) workspace `ws`. `ws.T` contains upper
triangular block reflectors which parameterize the elementary reflectors of
the factorization. The first dimension of `ws.T` sets the block size and it must
satisfy `1 <= size(ws.T, 1) <= min(size(A)...)`. The second dimension of `T` must equal the smallest
dimension of `A`, i.e. `size(ws.T, 2) == size(A, 2)`. If this is not the case and
`resize==true`, the workspace will automatically be resized to the appropriate dimensions.

`A` and `ws.T` are modified in-place.
"""
geqrt!(ws::QRWYWs, A::AbstractMatrix; kwargs...)

"""
    QRPivotedWs

Workspace to be used with the [`LinearAlgebra.QRPivoted`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.QRPivoted)
representation of the QR factorization which uses the [`LAPACK.geqp3!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = QRPivotedWs(A)
QRPivotedWs{Float64, Float64}
  work: 100-element Vector{Float64}
  rwork: 0-element Vector{Float64}
  τ: 2-element Vector{Float64}
  jpvt: 2-element Vector{Int64}

julia> t = QRPivoted(LAPACK.geqp3!(ws, A)...)
QRPivoted{Float64, Matrix{Float64}, Vector{Float64}, Vector{Int64}}
Q factor:
2×2 LinearAlgebra.QRPackedQ{Float64, Matrix{Float64}, Vector{Float64}}:
 -0.190022  -0.98178
 -0.98178    0.190022
R factor:
2×2 Matrix{Float64}:
 -6.31506  -3.67692
  0.0      -1.63102
permutation:
2-element Vector{Int64}:
 1
 2

julia> Matrix(t)
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3
```
"""
struct QRPivotedWs{T<:Number, RT<:AbstractFloat} <: Workspace
    work::Vector{T}
    rwork::Vector{RT}
    τ::Vector{T}
    jpvt::Vector{BlasInt}
end

for (geqp3, elty, relty) in ((:dgeqp3_, :Float64, :Float64),
                             (:sgeqp3_, :Float32, :Float32),
                             (:zgeqp3_, :ComplexF64, :Float64),
                             (:cgeqp3_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(ws::QRPivotedWs, A::StridedMatrix{$elty}; work=true)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            RldA = max(1, stride(A, 2))
            resize!(ws.jpvt, n)
            resize!(ws.τ, min(m, n))
            if work
                info = Ref{BlasInt}()
                if $elty <: Complex
                    resize!(ws.rwork, 2n)
                    ccall((@blasfunc($geqp3), liblapack), Cvoid,
                          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt}),
                          m, n, A, RldA, ws.jpvt, ws.τ, ws.work, -1, ws.rwork, info)
                else
                    ccall((@blasfunc($geqp3), liblapack), Cvoid,
                          (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                          m, n, A, RldA, ws.jpvt, ws.τ, ws.work, -1, info)
                end
                chklapackerror(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
        QRPivotedWs(A::StridedMatrix{$elty}) =
            resize!(QRPivotedWs(Vector{$elty}(undef, 1), $relty[], $elty[], BlasInt[]), A)

        function geqp3!(ws::QRPivotedWs{$elty}, A::AbstractMatrix{$elty}; resize=true)
            m, n = size(A)
            nws = length(ws.jpvt)
            minn =  min(m, n)
            if nws != n || minn != length(ws.τ)
                if resize
                    resize!(ws, A; work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, minn))
                end
            end
            lda = stride(A, 2)
            if lda == 0 # Early exit
                return A, ws.τ, ws.jpvt
            end
            info = Ref{BlasInt}()
            if $elty <: Complex
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                       Ptr{BlasInt}),
                      m, n, A, lda,
                      ws.jpvt, ws.τ, ws.work,
                      length(ws.work), ws.rwork, info)
            else
                ccall((@blasfunc($geqp3), liblapack), Cvoid,
                      (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}),
                      m, n, A, lda,
                      ws.jpvt, ws.τ, ws.work,
                      length(ws.work), info)
            end
            chklapackerror(info[])
            return A, ws.τ, ws.jpvt
        end
    end
end

"""
    geqp3!(ws, A; resize=true) -> (A, ws.τ, ws.jpvt)

Compute the pivoted `QR` factorization of `A`, `AP = QR` using BLAS level 3,
using the preallocated [`QRPivotedWs`](@ref) workspace `ws`.
`P` is a pivoting matrix, represented by `ws.jpvt`. `ws.τ` stores the elementary
reflectors. `ws.jpvt` must have length greater
than or equal to `n` if `A` is an `(m x n)` matrix and `ws.τ` must have length
greater than or equal to the smallest dimension of `A`. If this is not the case
and `resize == true` the workspace will be appropriately resized.

`A`, `ws.jpvt`, and `ws.τ` are modified in-place.
"""
geqp3!(ws::QRPivotedWs, A::AbstractMatrix; kwargs...)

"""
    QROrmWs

Workspace to be used with the [`LinearAlgebra.LAPACK.ormqr!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.LAPACK.ormqr!)
function. It requires the workspace of a `QR` or a `QRPivoted` previous factorization

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = QRPivotedWs(A)
QRPivotedWs{Float64, Float64}
  work: 100-element Vector{Float64}
  rwork: 0-element Vector{Float64}
  τ: 2-element Vector{Float64}
  jpvt: 2-element Vector{Int64}

julia> t = QRPivoted(LAPACK.geqp3!(ws, A)...)
QRPivoted{Float64, Matrix{Float64}, Vector{Float64}, Vector{Int64}}
Q factor:
2×2 LinearAlgebra.QRPackedQ{Float64, Matrix{Float64}, Vector{Float64}}:
 -0.190022  -0.98178
 -0.98178    0.190022
R factor:
2×2 Matrix{Float64}:
 -6.31506  -3.67692
  0.0      -1.63102
permutation:
2-element Vector{Int64}:
 1
 2

julia> Matrix(t)
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3
```
"""
struct QROrmWs{T<:Number} <: Workspace
    work::Vector{T}
    τ::Vector{T}
end

for (ormqr, orgqr, elty) in ((:dormqr_, :dorgqr_,  :Float64),
                             (:sormqr_, :sorgqr_, :Float32),
                             (:zunmqr_, :zungqr_, :ComplexF64),
                             (:cunmqr_, :cungqr_, :ComplexF32))
                      
    @eval begin
        function Base.resize!(ormws::QROrmWs, side::AbstractChar, trans::AbstractChar,
                              A::AbstractMatrix{$elty},
                              C::AbstractVecOrMat{$elty};
                              work = true)
            require_one_based_indexing(A, C)
            chktrans(trans)
            chkside(side)
            chkstride1(A, C)
            m, n = ndims(C) == 2 ? size(C) : (size(C, 1), 1)
            mA   = size(A, 1)
            k    = length(ormws.τ)
            if side == 'L' && m != mA
                throw(DimensionMismatch("for a left-sided multiplication, the first dimension of C, $m, must equal the second dimension of A, $mA"))
            end
            if side == 'R' && n != mA
                throw(DimensionMismatch("for a right-sided multiplication, the second dimension of C, $m, must equal the second dimension of A, $mA"))
            end
            if side == 'L' && k > m
                throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= m = $m"))
            end
            if side == 'R' && k > n
                throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= n = $n"))
            end
            if work
                info = Ref{BlasInt}()
                ccall((@blasfunc($ormqr), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ref{BlasInt}, Clong, Clong),
                      side, trans, m, n,
                      k, A, max(1, stride(A, 2)), ormws.τ,
                      C, max(1, stride(C, 2)), ormws.work, -1,
                      info, 1, 1)
                chklapackerror(info[])
                resize!(ormws.work, BlasInt(real(ormws.work[1])))
            end
            return ormws
        end

        QROrmWs(ws::Union{QRWs, QRPivotedWs}, side::AbstractChar, trans::AbstractChar,
                A::AbstractMatrix{$elty},
                C::AbstractVecOrMat{$elty}) = resize!(QROrmWs(Vector{$elty}(undef, 1), ws.τ), side, trans,
                                                      A, C)
        function ormqr!(ws::QROrmWs{$elty}, side::AbstractChar, trans::AbstractChar,
                          A::AbstractMatrix{$elty},
                          C::AbstractVecOrMat{$elty})
            require_one_based_indexing(A, C)
            chktrans(trans)
            chkside(side)
            chkstride1(A, C)
            m, n = ndims(C) == 2 ? size(C) : (size(C, 1), 1)
            mA   = size(A, 1)
            k    = length(ws.τ)
            if side == 'L' && m != mA
                throw(DimensionMismatch("for a left-sided multiplication, the first dimension of C, $m, must equal the second dimension of A, $mA"))
            end
            if side == 'R' && n != mA
                throw(DimensionMismatch("for a right-sided multiplication, the second dimension of C, $m, must equal the second dimension of A, $mA"))
            end
            if side == 'L' && k > m
                throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= m = $m"))
            end
            if side == 'R' && k > n
                throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= n = $n"))
            end
            info = Ref{BlasInt}()
            ccall((@blasfunc($ormqr), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ref{BlasInt}, Clong, Clong),
                  side, trans, m, n,
                  k, A, max(1, stride(A, 2)), ws.τ,
                  C, max(1, stride(C, 2)), ws.work, length(ws.work),
                  info, 1, 1)
            chklapackerror(info[])
            return C
        end
    end

    for elty2 in (eval(:(Transpose{$elty,<:StridedMatrix{$elty}})),
                  eval(:(Adjoint{$elty,<:StridedMatrix{$elty}})))
        @eval function ormqr!(ws::Union{QRWs{$elty}, QRPivotedWs{$elty}}, side::AbstractChar, trans::AbstractChar,
                              A::$elty2,
                              C::StridedMatrix{$elty})
            chktrans(trans)
            chkside(side)
            trans = trans == 'T' ? 'N' : 'T'
            return LAPACK.ormqr!(ws, side, trans, A.parent, C)
        end
    end
    @eval function orgqr!(ws::Union{QRWs{$elty}, QRPivotedWs{$elty}}, A::AbstractMatrix{$elty}, k::Integer = size(A, 2))
        require_one_based_indexing(A, ws.τ)
        chkstride1(A, ws.τ)
        m = size(A, 1)
        n = min(m, size(A, 2))
        if k > n
            throw(DimensionMismatch("invalid number of reflectors: k = $k should be <= n = $n"))
        end
        info  = Ref{BlasInt}()
        ccall((@blasfunc($orgqr), liblapack), Cvoid,
              (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
               Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
              m, n, k, A,
              max(1,stride(A,2)), ws.τ, ws.work, length(ws.work),
              info)
            chklapackerror(info[])
        if n < size(A,2)
            return A[:, 1:n]
        else
            return A
        end
    end
end

"""
    ormqr!(ws, side, trans, A, C) -> C

Computes `Q * C` (`trans = N`), `transpose(Q) * C` (`trans = T`), `adjoint(Q) * C`
(`trans = C`) for `side = L` or the equivalent right-sided multiplication
for `side = R` using `Q` from a `QR` factorization of `A` computed using
[`geqrf!`](@ref) in the case where `ws` is a [`QRWs`](@ref) or [`geqp3!`](@ref) when `ws` is a [`QRPivotedWs`](@ref).
Uses preallocated workspace `ws` and the factors are assumed to be stored in `ws.τ`.
`C` is overwritten.
"""
ormqr!(ws::Union{QRWs,QRPivotedWs}, side::AbstractChar, trans::AbstractChar, A::AbstractMatrix,
       C::AbstractVecOrMat)

"""
    orgqr!(ws, A, k = length(tau))

Explicitly finds the matrix `Q` of a `QR` factorization using the
factors stored in `ws.τ`, that were generated from calling 
[`geqrf!`](@ref) if `ws` is a [`QRWs`](@ref) or [`geqp3!`](@ref) if `ws` is a [`QRPivotedWs`](@ref).
`A` is overwritten by `Q`.
"""
orgqr!(ws::Union{QRWs, QRPivotedWs}, A::AbstractMatrix, k::Integer = size(A, 2))

