using Base: require_one_based_indexing
using LinearAlgebra: chkstride1
abstract type QR end

struct QRWs{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    τ::Vector{T}
end

for (geqrf, ormqr, elty) in (
    (:dgeqrf_, :dormqr_, :Float64),
    (:sgeqrf_, :sormqr_, :Float32),
    (:zgeqrf_, :zormqr_, :ComplexF64),
    (:cgeqrf_, :cormqr_, :ComplexF32),
)

    @eval begin
        function QRWs(A::StridedMatrix{$elty})
            m, n = size(A)
            lda = max(1, stride(A, 2))
            τ = Vector{$elty}(undef, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = -1
            info = Ref{BlasInt}()
            ccall(
                (@blasfunc($geqrf), liblapack),
                Nothing,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                m,
                n,
                A,
                lda,
                τ,
                work,
                lwork,
                info,
            )
            chklapackerror(info[])
            resize!(work, Int(real(work[1])))
            QRWs(work, info, τ)
        end

        function geqrf!(A::StridedMatrix{$elty}, ws::QRWs)
            m, n = size(A)
            lda = max(1, stride(A, 2))
            lwork = length(ws.work)
            ccall(
                (@blasfunc($geqrf), liblapack),
                Nothing,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                m,
                n,
                A,
                lda,
                ws.τ,
                ws.work,
                lwork,
                ws.info,
            )
            chklapackerror(ws.info[])
            return A, ws.τ 
        end

        t1 = StridedMatrix{$elty}
        t2 = Transpose{$elty,<:StridedMatrix}
        t3 = Adjoint{$elty,<:StridedMatrix}
    end
end

struct QRWsNew{T<:Number} <: QR
    work::Vector{T}
    info::Ref{BlasInt}
    T::StridedMatrix{T}
end

for (geqrt, ormqr, elty) in (
    (:dgeqrt_, :dormqr_, :Float64),
    (:sgeqrt_, :sormqr_, :Float32),
    (:zgeqrt_, :zormqr_, :ComplexF64),
    (:cgeqrt_, :cormqr_, :ComplexF32),
)

    @eval begin
        function QRWsNew(A::StridedMatrix{$elty}; blocksize = 36)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = BlasInt.(size(A))
            @assert n > 0 ArgumentError("Not a Matrix")
            m1 = min(m, n)
            nb = min(m1, blocksize)
            T = similar(A, nb, m1)
            
            work = Vector{$elty}(undef, nb*n)
            return QRWsNew(work, Ref{BlasInt}(), T)
        end
        
        function geqrt!(A::StridedMatrix{$elty}, ws::QRWsNew)
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            nb = size(ws.T, 1)
            lda = max(1, stride(A,2))
            work = ws.work
                
            ccall((@blasfunc($geqrt), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                 Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                 Ptr{BlasInt}),
                 m, n, nb, A,
                 lda, ws.T, max(1,stride(ws.T,2)), ws.work,
                 ws.info)
            chklapackerror(ws.info[])
            A, ws.T
        end
    end
end

for (geqrf, ormqr, elty) in ((:dgeqrf_, :dormqr_, :Float64), (:sgeqrf_, :sormqr_, :Float32))

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
            RldA = Ref{BlasInt}(max(1, stride(A, 2)))
            RldC = Ref{BlasInt}(max(1, stride(C, 2)))
            ccall(
                (@blasfunc($ormqr), liblapack),
                Nothing,
                (
                    Ref{UInt8},
                    Ref{UInt8},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                side,
                'N',
                m,
                n,
                k,
                A,
                RldA,
                ws.tau,
                C,
                RldC,
                ws.work,
                ws.lwork,
                ws.info,
            )
            chklapackerror(ws.info[])
        end
    end

    @eval begin
        t1 = Transpose{$elty,<:StridedMatrix{$elty}}
        t2 = Adjoint{$elty,<:StridedMatrix{$elty}}
    end

    for elty2 in (t1, t2)

        @eval begin
            function ormqr_core!(side::Char, A::$elty2, C::StridedMatrix{$elty}, ws::QR)
                mm, nn = size(C)
                m = Ref{BlasInt}(mm)
                n = Ref{BlasInt}(nn)
                k = Ref{BlasInt}(length(ws.tau))
                RldA = Ref{BlasInt}(max(1, stride(A.parent, 2)))
                RldC = Ref{BlasInt}(max(1, stride(C, 2)))
                ccall(
                    (@blasfunc($ormqr), liblapack),
                    Nothing,
                    (
                        Ref{UInt8},
                        Ref{UInt8},
                        Ref{BlasInt},
                        Ref{BlasInt},
                        Ref{BlasInt},
                        Ptr{$elty},
                        Ref{BlasInt},
                        Ptr{$elty},
                        Ptr{$elty},
                        Ref{BlasInt},
                        Ptr{$elty},
                        Ref{BlasInt},
                        Ref{BlasInt},
                    ),
                    side,
                    'T',
                    m,
                    n,
                    k,
                    A.parent,
                    RldA,
                    ws.tau,
                    C,
                    RldC,
                    ws.work,
                    ws.lwork,
                    ws.info,
                )
                chklapackerror(ws.info[])
            end
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
            ccall((@blasfunc($ormqr),  liblapack), Nothing,
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
                ccall((@blasfunc($ormqr),  liblapack), Nothing,
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

struct QrpWs{T<:Number} <: QR
    tau::Vector{T}
    jpvt::Vector{BlasInt}
    work::Vector{T}
    lwork::BlasInt
    info::Ref{BlasInt}
end

for (geqp3, elty) in (
    (:dgeqp3_, :Float64),
    (:sgeqp3_, :Float32),
    (:zgeqp3_, :ComplexF64),
    (:cgeqp3_, :ComplexF32),
)

    @eval begin

        function QrpWs(A::StridedMatrix{$elty})
            m, n = size(A)
            RldA = BlasInt(max(1, stride(A, 2)))
            jpvt = zeros(BlasInt, n)
            tau = Vector{$elty}(undef, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            ccall(
                (@blasfunc($geqp3), liblapack),
                Nothing,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                m,
                n,
                A,
                RldA,
                jpvt,
                tau,
                work,
                lwork,
                info,
            )
            chklapackerror(info[])
            lwork = BlasInt(real(work[1]))
            work = resize!(work, lwork)
            QrpWs(tau, jpvt, work, lwork, info)
        end

        function geqp3!(A::StridedMatrix{$elty}, ws::QrpWs)
            m, n = size(A)
            RldA = BlasInt(max(1, stride(A, 2)))
            ccall(
                (@blasfunc($geqp3), liblapack),
                Nothing,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Ptr{$elty},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ref{BlasInt},
                ),
                m,
                n,
                A,
                RldA,
                ws.jpvt,
                ws.tau,
                ws.work,
                ws.lwork,
                ws.info,
            )
            chklapackerror(ws.info[])
        end

        t1 = StridedMatrix{$elty}
        t2 = Transpose{$elty,<:StridedMatrix}
        t3 = Adjoint{$elty,<:StridedMatrix}
    end
end

