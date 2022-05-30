using Base: require_one_based_indexing
using LinearAlgebra.LAPACK: chkstride1, chktrans, chkside
abstract type QR end


# We should check for bounds etc when reusing caches
struct QRWs{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    τ::Vector{T}
end

for (geqrf, ormqr, elty) in ((:dgeqrf_, :dormqr_, :Float64),
                             (:sgeqrf_, :sormqr_, :Float32),
                             (:zgeqrf_, :zormqr_, :ComplexF64),
                             (:cgeqrf_, :cormqr_, :ComplexF32))
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

        function geqrf!(A::StridedMatrix{$elty}, ws::QRWs)
            m, n = size(A)
            lda = max(1, stride(A, 2))
            lwork = length(ws.work)
            ccall((@blasfunc($geqrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                   Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, lda, ws.τ, ws.work, lwork, ws.info)
            chklapackerror(ws.info[])
            return A, ws.τ
        end
        # t1 = StridedMatrix{$elty}
        # t2 = Transpose{$elty,<:StridedMatrix}
        # t3 = Adjoint{$elty,<:StridedMatrix}
    end
end

# Should we add all functions similar to Base?
for (geqrf, ormqr, elty) in ((:dgeqrf_, :dormqr_, :Float64), (:sgeqrf_, :sormqr_, :Float32))
    @eval begin
        function ormqr!(side::AbstractChar, trans::AbstractChar, A::AbstractMatrix{$elty},
                        C::AbstractVecOrMat{$elty}, ws::QRWs{$elty})
            require_one_based_indexing(A, C)
            chktrans(trans)
            chkside(side)
            chkstride1(A, C)
            m,n = ndims(C) == 2 ? size(C) : (size(C, 1), 1)
            mA  = size(A, 1)
            k   = length(ws.τ)
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
                  k, A, max(1,stride(A,2)), ws.τ,
                  C, max(1, stride(C,2)), ws.work, length(ws.work),
                  ws.info, 1, 1)
            chklapackerror(ws.info[])
            return C
        end
    end

    for elty2 in (eval(:(Transpose{$elty,<:StridedMatrix{$elty}})),
                  eval(:(Adjoint{$elty,<:StridedMatrix{$elty}})))
        @eval begin
            function ormqr!(side::AbstractChar, trans::AbstractChar, A::$elty2, C::StridedMatrix{$elty}, ws::QRWs{$elty})
                chktrans(trans)
                chkside(side)
                trans = trans == 'T' ? 'N' : 'T'
                ormqr!(side, trans, A.parent, C, ws)
            end
        end
    end
end


struct QRWsWY{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    T::StridedMatrix{T}
end

for (geqrt, ormqr, elty) in ((:dgeqrt_, :dormqr_, :Float64),
                             (:sgeqrt_, :sormqr_, :Float32),
                             (:zgeqrt_, :zormqr_, :ComplexF64),
                             (:cgeqrt_, :cormqr_, :ComplexF32))
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
            nb = size(ws.T, 1)
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

#=
for (geqrf, ormqr, elty) in
    ((:zgeqrf_, :zormqr_, :ComplexF64), (:cgeqrf_, :cormqr_, :ComplexF32))

    @eval begin
        function ormqr_core!(
            side::Char,
            A::StridedMatrix{$elty},
            C::StridedMatrix{$elty},
            ws::QR,
        )
            mm, nn = size(C)
            m = Ref{BlasInt}(mm)
            n = Ref{BlasInt}(nn)
            k = Ref{BlasInt}(length(ws.tau))
            RldA = Ref{BlasInt}(max(1,stride(A,2)))
            RldC = Ref{BlasInt}(max(1,stride(C,2)))
            ccall((@blasfunc($ormqr),  liblapac Cvoid,
                  (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                   Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                  side, 'N', m, n, k, A, RldA, ws.tau, C, RldC, ws.work, ws.lwork, ws.info)
            chklapackerror(ws.info[])
        end
    end

    @eval begin
        t1 = Transpose{$elty,<:StridedMatrix{$elty}}
        t2 = Adjoint{$elty,<:StridedMatrix{$elty}}
    end

    for (elty2, transchar) in ((t2, 'T'), (t3, 'C'))

        @eval begin
            function ormqr_core!(side::Char, A::$elty2, C::StridedMatrix{$elty}, ws::QR)
                mm, nn = size(C)
                m = Ref{BlasInt}(mm)
                n = Ref{BlasInt}(nn)
                k = Ref{BlasInt}(length(ws.tau))
                RldA = Ref{BlasInt}(max(1,stride(A.parent,2)))
                RldC = Ref{BlasInt}(max(1,stride(C,2)))
                ccall((@blasfunc($ormqr),  liblapac Cvoid,
                      (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                       Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                      side, $transchar, m, n, k, A.parent, RldA,
                      ws.tau, C, RldC, ws.work, ws.lwork, ws.info)
                chklapackerror(ws.info[])
            end
        end
    end    
end 
=#

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
            if length(ws.τ) != min(m,n)
                throw(DimensionMismatch("tau has length $(length(ws.τ)), but needs length $(min(m,n))"))
            end
            if length(ws.jpvt) != n
                throw(DimensionMismatch("jpvt has length $(length(ws.jpvt)), but needs length $n"))
            end
            lda = stride(A,2)
            if lda == 0
                return A, ws.τ, ws.jpvt
            end # Early exit
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

        # t1 = StridedMatrix{$elty}
        # t2 = Transpose{$elty,<:StridedMatrix}
        # t3 = Adjoint{$elty,<:StridedMatrix}
    end
end
