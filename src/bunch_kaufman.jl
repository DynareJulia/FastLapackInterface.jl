using LinearAlgebra.LAPACK: chkuplo
import LinearAlgebra.LAPACK: sytrf!, sytrf_rook!

"""
"""
struct BunchKaufmanWs{T} <: Workspace
    work::Vector{T}
    ipiv::Vector{BlasInt}
end

for (sytrfs,  elty) in
    (((:dsytrf_,:dsytrf_rook_),:Float64),
     ((:ssytrf_,:ssytrf_rook_),:Float32),
     ((:zsytrf_,:zsytrf_rook_),:ComplexF64),
     ((:csytrf_,:csytrf_rook_),:ComplexF32))  
    @eval function BunchKaufmanWs(A::AbstractMatrix{$elty})
        chkstride1(A)
        n = checksquare(A)
        ipiv  = similar(A, BlasInt, n)
        if n == 0
            return BunchKaufmanWs($elty[], ipiv)
        end
        work  = Vector{$elty}(undef, 1)
        lwork = BlasInt(-1)
        info  = Ref{BlasInt}()
        ccall((@blasfunc($(sytrfs[1])), liblapack), Cvoid,
              (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
               Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
              'U', n, A, stride(A,2), ipiv, work, lwork, info, 1)
        chkargsok(info[])
        resize!(work, BlasInt(real(work[1])))
        return BunchKaufmanWs(work, ipiv)
    end
    for (sytrf, fn) in zip(sytrfs, (:sytrf!, :sytrf_rook!))
        @eval function $fn(ws::BunchKaufmanWs{$elty}, uplo::AbstractChar, A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
            chkuplo(uplo)
            lipiv = length(ws.ipiv)
            @assert n <= lipiv "Workspace was allocated for matrices of maximum size ($lipiv, $lipiv)." 
            if n == 0
                return A, ws.ipiv, zero(BlasInt)
            end
            info  = Ref{BlasInt}()
            ccall((@blasfunc($sytrf), liblapack), Cvoid,
                  (Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                   Ptr{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Clong),
                  uplo, n, A, stride(A,2), ws.ipiv, ws.work, length(ws.work), info, 1)
            chkargsok(info[])
            return A, ws.ipiv, info[]
        end
    end
end
