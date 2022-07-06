# TODO: LinearAlgebra.qr! method that works with any workspace
import LinearAlgebra.LAPACK: geqrf!, ormqr!, geqrt!, geqp3!

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
2×2 QRPackedQ{Float64, Matrix{Float64}, Vector{Float64}}:
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
struct QRWs{T<:Number}
    work::Vector{T}
    τ::Vector{T}
end

Base.length(ws::QRWs) = length(ws.τ)

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::QRWs)
    summary(io, ws)
    println(io)
    print(io, "work: ")
    summary(io, ws.work)
    println(io)
    print(io, "τ: ")
    return summary(io, ws.τ)
end

for (geqrf, elty) in ((:dgeqrf_, :Float64),
                      (:sgeqrf_, :Float32),
                      (:zgeqrf_, :ComplexF64),
                      (:cgeqrf_, :ComplexF32))
    @eval begin
        function QRWs(A::StridedMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            τ = zeros($elty, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = -1
            info = Ref{BlasInt}()
            ccall((@blasfunc($geqrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, τ, work, lwork, info)
            chklapackerror(info[])
            resize!(work, BlasInt(real(work[1])))
            return QRWs(work, τ)
        end

        function geqrf!(ws::QRWs, A::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            if length(ws) != min(m, n)
                throw(DimensionMismatch("Allocated workspace has length $(length(ws)), but needs length $(min(m,n))"))
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

for (ormqr, elty) in ((:dormqr_, :Float64),
                      (:sormqr_, :Float32))
    @eval begin
        function ormqr!(ws::QRWs{$elty}, side::AbstractChar, trans::AbstractChar,
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
        @eval begin
            function ormqr!(ws::QRWs{$elty}, side::AbstractChar, trans::AbstractChar, A::$elty2,
                            C::StridedMatrix{$elty})
                chktrans(trans)
                chkside(side)
                trans = trans == 'T' ? 'N' : 'T'
                return LAPACK.ormqr!(ws, side, trans, A.parent, C)
            end
        end
    end
end

"""
    geqrf!(ws, A) -> (A, ws.τ)

Compute the `QR` factorization of `A`, `A = QR`, using previously allocated [`QRWs`](@ref) workspace `ws`.
`ws.τ` contains scalars which parameterize the elementary reflectors of the factorization.
`ws.τ` must have length greater than or equal to the smallest dimension of `A`.

`A` and `ws.τ` modified in-place.
"""
geqrf!(ws::QRWs, A::AbstractMatrix)

"""
    ormqr!(ws, side, trans, A, C) -> C

Computes `Q * C` (`trans = N`), `transpose(Q) * C` (`trans = T`), `adjoint(Q) * C`
(`trans = C`) for `side = L` or the equivalent right-sided multiplication
for `side = R` using `Q` from a `QR` factorization of `A` computed using
`geqrf!`.
Uses preallocated workspace `ws` and the factors are assumed to be stored in `ws.τ`.
`C` is overwritten.
"""
ormqr!(ws::QRWs, side::AbstractChar, trans::AbstractChar, A::AbstractMatrix,
       C::AbstractVecOrMat)

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
blocksize: 2
work: 4-element Vector{Float64}
T: 2×2 Matrix{Float64}

julia> t = QRCompactWY(LAPACK.geqrt!(ws, A)...)
QRCompactWY{Float64, Matrix{Float64}, Matrix{Float64}}
Q factor:
2×2 QRCompactWYQ{Float64, Matrix{Float64}, Matrix{Float64}}:
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
struct QRWYWs{R<:Number,MT<:StridedMatrix{R}}
    work::Vector{R}
    T::MT
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::QRWYWs)
    summary(io, ws)
    println(io)
    println(io, "blocksize: $(size(ws.T, 1))")
    print(io, "work: ")
    summary(io, ws.work)
    println(io)
    print(io, "T: ")
    return summary(io, ws.T)
end

for (geqrt, elty) in ((:dgeqrt_, :Float64),
                      (:sgeqrt_, :Float32),
                      (:zgeqrt_, :ComplexF64),
                      (:cgeqrt_, :ComplexF32))
    @eval begin
        function QRWYWs(A::StridedMatrix{$elty}; blocksize = 36)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = BlasInt.(size(A))
            @assert n > 0 ArgumentError("Not a Matrix")
            m1 = min(m, n)
            nb = min(m1, blocksize)
            T = zeros($elty, nb, m1)

            work = zeros($elty, nb * n)
            return QRWYWs(work, T)
        end

        function geqrt!(ws::QRWYWs, A::AbstractMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            minmn = min(m, n)
            nb = size(ws.T, 1)
            if nb > minmn
                throw(ArgumentError("Allocated workspace block size $nb > $minmn too large."))
            end
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
end

"""
    geqrt!(ws, A) -> (A, ws.T)

Compute the blocked `QR` factorization of `A`, `A = QR`, using a preallocated [`QRWYWs`](@ref) workspace `ws`. `ws.T` contains upper
triangular block reflectors which parameterize the elementary reflectors of
the factorization. The first dimension of `ws.T` sets the block size and it must
satisfy `1 <= size(ws.T, 1) <= min(size(A)...)`. The second dimension of `T` must equal the smallest
dimension of `A`, i.e. `size(ws.T, 2) == size(A, 2)`.

`A` and `ws.T` are modified in-place.
"""
geqrt!(ws::QRWYWs, A::AbstractMatrix)

"""
    QRpWs

Workspace to be used with the [`LinearAlgebra.QRPivoted`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.QRPivoted)
representation of the QR factorization which uses the [`LAPACK.geqp3!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = QRpWs(A)
QRpWs{Float64}
work: 100-element Vector{Float64}
τ: 2-element Vector{Float64}
jpvt: 2-element Vector{Int64}

julia> t = QRPivoted(LAPACK.geqp3!(ws, A)...)
QRPivoted{Float64, Matrix{Float64}, Vector{Float64}, Vector{Int64}}
Q factor:
2×2 QRPackedQ{Float64, Matrix{Float64}, Vector{Float64}}:
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
struct QRpWs{T<:Number}
    work::Vector{T}
    τ::Vector{T}
    jpvt::Vector{BlasInt}
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::QRpWs)
    summary(io, ws)
    println(io)
    print(io, "work: ")
    summary(io, ws.work)
    println(io)
    print(io, "τ: ")
    summary(io, ws.τ)
    println(io)
    print(io, "jpvt: ")
    return summary(io, ws.jpvt)
end

for (geqp3, elty) in ((:dgeqp3_, :Float64),
                      (:sgeqp3_, :Float32),
                      (:zgeqp3_, :ComplexF64),
                      (:cgeqp3_, :ComplexF32))
    @eval begin
        function QRpWs(A::StridedMatrix{$elty})
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            RldA = max(1, stride(A, 2))
            jpvt = zeros(BlasInt, n)
            τ = zeros($elty, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = -1
            info = Ref{BlasInt}()
            ccall((@blasfunc($geqp3), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, RldA, jpvt, τ, work, lwork, info)
            chklapackerror(info[])
            work = resize!(work, BlasInt(real(work[1])))
            return QRpWs(work, τ, jpvt)
        end

        function geqp3!(ws::QRpWs{$elty}, A::AbstractMatrix{$elty})
            m, n = size(A)
            if length(ws.τ) != min(m, n)
                throw(DimensionMismatch("τ  has length $(length(ws.τ)), but needs length $(min(m,n))"))
            end
            if length(ws.jpvt) != n
                throw(DimensionMismatch("jpvt has length $(length(ws.jpvt)), but needs length $n"))
            end
            lda = stride(A, 2)
            if lda == 0 # Early exit
                return A, ws.τ, ws.jpvt
            end
            info = Ref{BlasInt}()
            ccall((@blasfunc($geqp3), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}),
                  m, n, A, lda,
                  ws.jpvt, ws.τ, ws.work,
                  length(ws.work), info)
            chklapackerror(info[])
            return A, ws.τ, ws.jpvt
        end
    end
end

"""
    geqp3!(ws, A) -> (A, ws.τ, ws.jpvt)

Compute the pivoted `QR` factorization of `A`, `AP = QR` using BLAS level 3,
using the preallocated [`QRpWs`](@ref) workspace `ws`.
`P` is a pivoting matrix, represented by `ws.jpvt`. `ws.τ` stores the elementary
reflectors. `ws.jpvt` must have length greater
than or equal to `n` if `A` is an `(m x n)` matrix and `ws.τ` must have length
greater than or equal to the smallest dimension of `A`.

`A`, `ws.jpvt`, and `ws.τ` are modified in-place.
"""
geqp3!(ws::QRpWs, A::AbstractMatrix)
