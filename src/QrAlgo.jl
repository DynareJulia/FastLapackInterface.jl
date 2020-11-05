module QrAlgo

using LinearAlgebra
import LinearAlgebra: BlasInt
import LinearAlgebra.BLAS: @blasfunc
import LinearAlgebra.LAPACK: liblapack, chklapackerror

export QrWs, QrpWs, geqrf_core!, geqp3!, ormqr_core!

abstract type QR end

struct QrWs{T <: Number} <: QR
    tau::Vector{T}
    work::Vector{T}
    lwork::Ref{BlasInt}
    info::Ref{BlasInt}
end

for (geqrf, ormqr, elty) in
    ((:dgeqrf_, :dormqr_, :Float64),
     (:sgeqrf_, :sormqr_, :Float32),
     (:zgeqrf_, :zormqr_, :ComplexF64),
     (:cgeqrf_, :cormqr_, :ComplexF32))

    @eval begin

        function QrWs(A::StridedMatrix{T}) where T <: $elty
            nn, mm = size(A)
            m = Ref{BlasInt}(mm)
            n = Ref{BlasInt}(nn)
            RldA = Ref{BlasInt}(max(1,stride(A,2)))
            tau = Vector{T}(undef, min(nn,mm))
            work = Vector{T}(undef, 1)
            lwork = Ref{BlasInt}(-1)
            info = Ref{BlasInt}(0)
            ccall((@blasfunc($geqrf), liblapack), Nothing,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{T}, Ref{BlasInt},
                   Ptr{T}, Ptr{T}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, RldA, tau, work, lwork, info)
            chklapackerror(info[])
            lwork = Ref{BlasInt}(real(work[1]))
            work = Array{T}(undef, lwork[])
            QrWs(tau, work, lwork, info)
        end

        function geqrf_core!(A::StridedMatrix{$elty}, ws::QrWs)
            mm,nn = size(A)
            m = Ref{BlasInt}(mm)
            n = Ref{BlasInt}(nn)
            RldA = Ref{BlasInt}(max(1,stride(A,2)))
            ccall((@blasfunc($geqrf), liblapack), Nothing,
                  (Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                   Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                  m,n,A,RldA,ws.tau,ws.work,ws.lwork,ws.info)
            chklapackerror(ws.info[])
        end

        t1 = StridedMatrix{$elty}
        t2 = Transpose{$elty, <: StridedMatrix}
        t3 = Adjoint{$elty, <: StridedMatrix}
    end
end

for (geqrf, ormqr, elty) in
    ((:dgeqrf_, :dormqr_, :Float64),
     (:sgeqrf_, :sormqr_, :Float32))

    @eval begin
        function ormqr_core!(side::Char, A::StridedMatrix{$elty},
                              C::StridedMatrix{$elty}, ws::QR)
            mm,nn = size(C)
            m = Ref{BlasInt}(mm)
            n = Ref{BlasInt}(nn)
            k = Ref{BlasInt}(length(ws.tau))
            RldA = Ref{BlasInt}(max(1,stride(A,2)))
            RldC = Ref{BlasInt}(max(1,stride(C,2)))
            ccall((@blasfunc($ormqr), liblapack), Nothing,
                  (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                   Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                  side, 'N', m, n, k, A, RldA, ws.tau, C, RldC, ws.work, ws.lwork, ws.info)
            chklapackerror(ws.info[])
        end
    end

    @eval begin
        t1 = Transpose{$elty, <: StridedMatrix{$elty}}
        t2 = Adjoint{$elty, <: StridedMatrix{$elty}}
    end
    
    for elty2 in (t1, t2)
        
        @eval begin
            function ormqr_core!(side::Char, A::$elty2,
                                  C::StridedMatrix{$elty}, ws::QR)
                mm,nn = size(C)
                m = Ref{BlasInt}(mm)
                n = Ref{BlasInt}(nn)
                k = Ref{BlasInt}(length(ws.tau))
                RldA = Ref{BlasInt}(max(1,stride(A.parent,2)))
                RldC = Ref{BlasInt}(max(1,stride(C,2)))
                ccall((@blasfunc($ormqr), liblapack), Nothing,
                      (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                       Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                      side, 'T', m, n, k, A.parent, RldA, ws.tau, C, RldC, ws.work, ws.lwork, ws.info)
                chklapackerror(ws.info[])
            end
        end
    end    
end 
                      
for (geqrf, ormqr, elty) in
    ((:zgeqrf_, :zormqr_, :ComplexF64),
     (:cgeqrf_, :cormqr_, :ComplexF32))

    @eval begin
        function ormqr_core!(side::Char, A::StridedMatrix{$elty},
                              C::StridedMatrix{$elty}, ws::QR)
            mm,nn = size(C)
            m = Ref{BlasInt}(mm)
            n = Ref{BlasInt}(nn)
            k = Ref{BlasInt}(length(ws.tau))
            RldA = Ref{BlasInt}(max(1,stride(A,2)))
            RldC = Ref{BlasInt}(max(1,stride(C,2)))
            ccall((@blasfunc($ormqr), liblapack), Nothing,
                  (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                   Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                  side, 'N', m, n, k, A, RldA, ws.tau, C, RldC, ws.work, ws.lwork, ws.info)
            chklapackerror(ws.info[])
        end
    end

    @eval begin
        t1 = Transpose{$elty, <: StridedMatrix{$elty}}
        t2 = Adjoint{$elty, <: StridedMatrix{$elty}}
    end
    
    for (elty2, transchar) in
         ((t2, 'T'),
          (t3, 'C'))

        @eval begin
            function ormqr_core!(side::Char, A::$elty2,
                                  C::StridedMatrix{$elty}, ws::QR)
                mm,nn = size(C)
                m = Ref{BlasInt}(mm)
                n = Ref{BlasInt}(nn)
                k = Ref{BlasInt}(length(ws.tau))
                RldA = Ref{BlasInt}(max(1,stride(A.parent,2)))
                RldC = Ref{BlasInt}(max(1,stride(C,2)))
                ccall((@blasfunc($ormqr), liblapack), Nothing,
                      (Ref{UInt8},Ref{UInt8},Ref{BlasInt},Ref{BlasInt},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},
                       Ptr{$elty},Ptr{$elty},Ref{BlasInt},Ptr{$elty},Ref{BlasInt},Ref{BlasInt}),
                      side, $transchar, m, n, k, A.parent, RldA, ws.tau, C, RldC, ws.work, ws.lwork, ws.info)
                chklapackerror(ws.info[])
            end
        end
    end    
end 
                      
struct QrpWs{T <: Number}  <: QR
    tau::Vector{T}
    jpvt::Vector{BlasInt}
    work::Vector{T}
    lwork::BlasInt
    info::Ref{BlasInt}
end

for (geqp3, elty) in
    ((:dgeqp3_, :Float64),
     (:sgeqp3_, :Float32),
     (:zgeqp3_, :ComplexF64),
     (:cgeqp3_, :ComplexF32))

    @eval begin

        function QrpWs(A::StridedMatrix{$elty})
            m, n = size(A)
            RldA = BlasInt(max(1,stride(A,2)))
            jpvt = zeros(BlasInt, n)
            tau = Vector{$elty}(undef, min(m, n))
            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            ccall((@blasfunc($geqp3), liblapack), Nothing,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, RldA, jpvt, tau, work, lwork, info)
            chklapackerror(info[])
            lwork = BlasInt(real(work[1]))
            @show lwork
            @show m
            @show n
            @show RldA
            @show info[]
            work = resize!(work, lwork)
            QrpWs(tau, jpvt, work, lwork, info)
        end

        function geqp3!(A::StridedMatrix{$elty}, ws::QrpWs)
            m, n = size(A)
            RldA = BlasInt(max(1,stride(A,2)))
            ccall((@blasfunc($geqp3), liblapack), Nothing,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt}),
                  m, n, A, RldA, ws.jpvt, ws.tau, ws.work, ws.lwork, ws.info)
            chklapackerror(ws.info[])
        end

        t1 = StridedMatrix{$elty}
        t2 = Transpose{$elty, <: StridedMatrix}
        t3 = Adjoint{$elty, <: StridedMatrix}
    end
end

end
