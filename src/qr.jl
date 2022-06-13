using Base: require_one_based_indexing
using LinearAlgebra.LAPACK: chkstride1, chktrans, chkside
abstract type QR end

struct QRWs{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    τ::Vector{T}
end

Base.length(ws::QRWs) = length(ws.τ)

for (geqrf, elty) in ((:dgeqrf_, :Float64),
                      (:sgeqrf_, :Float32),
                      (:zgeqrf_, :ComplexF64),
                      (:cgeqrf_, :ComplexF32))
    @eval begin
        function QRWs(A::StridedMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            τ = Vector{$elty}(undef, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = -1
            info = Ref{BlasInt}()
            ccall((@blasfunc($geqrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, lda, τ, work, lwork, info)
            chklapackerror(info[])
            resize!(work, BlasInt(real(work[1])))
            return QRWs(work, info, τ)
        end

        function geqrf!(A::AbstractMatrix{$elty}, ws::QRWs)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            if length(ws) != min(m, n)
                throw(DimensionMismatch("Allocated workspace has length $(length(ws)), but needs length $(min(m,n))"))
            end
            lda = max(1, stride(A, 2))
            lwork = length(ws.work)
            ccall((@blasfunc($geqrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, lda, ws.τ, ws.work, lwork, ws.info)
            chklapackerror(ws.info[])
            return A, ws.τ
        end
    end
end

for (ormqr, elty) in ((:dormqr_, :Float64),
                      (:sormqr_, :Float32))
    @eval begin
        function ormqr!(side::AbstractChar, trans::AbstractChar, A::AbstractMatrix{$elty},
                        C::AbstractVecOrMat{$elty}, ws::QRWs{$elty})
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
            ccall((@blasfunc($ormqr), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Clong, Clong),
                  side, trans, m, n,
                  k, A, max(1, stride(A, 2)), ws.τ,
                  C, max(1, stride(C, 2)), ws.work, length(ws.work),
                  ws.info, 1, 1)
            chklapackerror(ws.info[])
            return C
        end
    end

    for elty2 in (eval(:(Transpose{$elty,<:StridedMatrix{$elty}})),
                  eval(:(Adjoint{$elty,<:StridedMatrix{$elty}})))
        @eval begin
            function ormqr!(side::AbstractChar, trans::AbstractChar, A::$elty2,
                            C::StridedMatrix{$elty}, ws::QRWs{$elty})
                chktrans(trans)
                chkside(side)
                trans = trans == 'T' ? 'N' : 'T'
                return ormqr!(side, trans, A.parent, C, ws)
            end
        end
    end
end

struct QRWsWY{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    T::StridedMatrix{T}
end

for (geqrt, elty) in ((:dgeqrt_, :Float64),
                      (:sgeqrt_, :Float32),
                      (:zgeqrt_, :ComplexF64),
                      (:cgeqrt_, :ComplexF32))
    @eval begin
        function QRWsWY(A::StridedMatrix{$elty}; blocksize = 36)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = BlasInt.(size(A))
            @assert n > 0 ArgumentError("Not a Matrix")
            m1 = min(m, n)
            nb = min(m1, blocksize)
            T = similar(A, nb, m1)

            work = Vector{$elty}(undef, nb * n)
            return QRWsWY(work, Ref{BlasInt}(), T)
        end

        function geqrt!(A::StridedMatrix{$elty}, ws::QRWsWY)
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

            ccall((@blasfunc($geqrt), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{BlasInt}),
                  m, n, nb, A,
                  lda, ws.T, max(1, stride(ws.T, 2)), ws.work,
                  ws.info)
            chklapackerror(ws.info[])
            return A, ws.T
        end
    end
end

struct QRpWs{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    τ::Vector{T}
    jpvt::Vector{BlasInt}
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
            τ = Vector{$elty}(undef, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            ccall((@blasfunc($geqp3), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                   Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, RldA, jpvt, τ, work, lwork, info)
            chklapackerror(info[])
            work = resize!(work, BlasInt(real(work[1])))
            return QRpWs(work, info, τ, jpvt)
        end

        function geqp3!(A::AbstractMatrix{$elty}, ws::QRpWs{$elty})
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
            lwork = BlasInt(-1)
            ccall((@blasfunc($geqp3), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}),
                  m, n, A, lda,
                  ws.jpvt, ws.τ, ws.work,
                  length(ws.work), ws.info)
            chklapackerror(ws.info[])
            return A, ws.τ, ws.jpvt
        end
    end
end
