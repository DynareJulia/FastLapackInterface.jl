# Workspaces for Eigenvalue decomposition
using LinearAlgebra.LAPACK: chkfinite
import LinearAlgebra.LAPACK: geevx!, syevr!


"""
    EigenWs

Workspace for [`LinearAlgebra.Eigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.Eigen)
factorization using the [`LAPACK.geevx!`](@ref) function.

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

julia> t = QR(LAPACK.geqrf!(A, ws)...)
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
struct EigenWs{T, MT <: AbstractMatrix{T}, RT<:AbstractFloat}
    work::Vector{T}
    rwork::Vector{RT} # Can be rwork if T <: Complex or WI if T <: Float64
    VL::MT
    VR::MT
    W::Vector{T} # Can be W or WR
    scale::Vector{RT}
    iwork::Vector{BlasInt}
    rconde::Vector{RT}
    rcondv::Vector{RT}
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::EigenWs)
    summary(io, ws)
    println(io)
    for f in fieldnames(EigenWs)
        print(io, "$f: ")
        summary(io, getfield(ws, f))
    end
end

# fns = fieldnames(EigenWs)
# @eval Base.iterate(ws::EigenWs) = (getfield(ws, $(fns[1])), Val(fns[2]))
# for i in 2:length(fns) - 1
#     f = fns[i]
#     @eval Base.iterate(ws::EigenWs, ::Val{$f}) = (getfield(ws, $f), Val(:($(fns[i+1]))))
# end
# @eval Base.iterate(ws::EigenWs, ::Val{$(fns[end])}) = (getfield(ws, $(fns[end])), Val(:done))
# Base.iterate(::EigenWs, ::Val{:done})    = nothing
Base.iterate(ws::EigenWs)                = (ws.work, Val(:rwork))
Base.iterate(ws::EigenWs, ::Val{:rwork}) = (ws.rwork, Val(:VL))
Base.iterate(ws::EigenWs, ::Val{:VL})    = (ws.VL, Val(:VR))
Base.iterate(ws::EigenWs, ::Val{:VR})    = (ws.VR, Val(:W))
Base.iterate(ws::EigenWs, ::Val{:W})     = (ws.W, Val(:scale))
Base.iterate(ws::EigenWs, ::Val{:scale}) = (ws.scale, Val(:iwork))
Base.iterate(ws::EigenWs, ::Val{:iwork}) = (ws.iwork, Val(:rconde))
Base.iterate(ws::EigenWs, ::Val{:rconde}) = (ws.rconde, Val(:rcondv))
Base.iterate(ws::EigenWs, ::Val{:rcondv}) = (ws.rcondv, Val(:done))
Base.iterate(::EigenWs, ::Val{:done})    = nothing

for (geevx, elty, relty) in
    ((:dgeevx_,:Float64,:Float64),
     (:sgeevx_,:Float32,:Float32),
     (:zgeevx_,:ComplexF64,:Float64),
     (:cgeevx_,:ComplexF32,:Float32))
    @eval begin
       
        function EigenWs(A::AbstractMatrix{$elty}; lvecs = false, rvecs = true, sense = false)
            chkstride1(A)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs

            if sense
                lvecs = true
                rvecs = true
                S = 'B'
            else
                S = 'N'
            end
            
            jobvl = lvecs ? 'V' : 'N'
            jobvr = rvecs ? 'V' : 'N'
            
            VL    = similar(A, $elty, (lvecs ? n : 0, n))
            VR    = similar(A, $elty, (rvecs ? n : 0, n))
            cmplx = eltype(A) <: Complex
            if cmplx
                W     = similar(A, $elty, n)
                rwork = similar(A, $relty, 2n)
            else
                W     = similar(A, $elty, n)
                rwork = similar(A, $elty, n)
            end
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            scale = zeros($relty, n)
            abnrm = Ref{$relty}()
            
            rconde = zeros($relty, sense ? n : 0)
            rcondv = zeros($relty, sense ? n : 0)

            iwork = zeros(BlasInt, sense ? 2n - 1 : 0)
            
            if cmplx
                      
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Clong, Clong, Clong, Clong),
                       'N', jobvl, jobvr, S,
                       n, A, max(1,stride(A,2)), W,
                       VL,n, VR, n,
                       ilo, ihi, scale, abnrm,
                       rconde, rcondv, work, lwork,
                       rwork, info, 1, 1, 1, 1)
                       
            else
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt},
                       Clong, Clong, Clong, Clong),
                       'N', jobvl, jobvr, 'N',
                       n, A, max(1,stride(A,2)), W,
                       rwork, VL, n, VR,
                       n, ilo, ihi, scale,
                       abnrm, rconde, rcondv, work,
                       lwork, iwork, info,
                       1, 1, 1, 1)
            end
            chklapackerror(info[])
            resize!(work, BlasInt(real(work[1])))
            return EigenWs(work, rwork, VL, VR, W, scale, iwork, rconde, rcondv)
        end
        
        function geevx!(balanc::AbstractChar, jobvl::AbstractChar, jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty}, ws::EigenWs)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            if balanc ∉ ('N', 'P', 'S', 'B')
                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
            end
            if sense ∉ ('N','E','V','B')
                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
            end
            # work, rwork, VL, VR, W, scale, iwork, rconde, rcondv = ws
            ldvl = size(ws.VL, 1)
            ldvr = size(ws.VR, 1)

            if sense ∈ ('E', 'B')
                if jobvl != 'V' || jobvr != 'V'
                    throw(ArgumentError("If sense = $sense it is required that jobvl = 'V' (is $jobvl) and jobvr = 'V' (is $jobvr)."))
                elseif size(ws.iwork, 1) == 0
                    throw(ArgumentError("Workspace was created without support for sense,\nrecreate with EigenWs(A, sense = true)"))
                end
            end
            @assert jobvl != 'V' || ldvl > 0 "Workspace was created without support for left eigenvectors,\nrecreate with EigenWs(A, lvecs = true)"
            @assert jobvr != 'V' || ldvr > 0 "Workspace was created without support for right eigenvectors,\nrecreate with EigenWs(A, rvecs = true)"
            
            abnrm = Ref{$relty}()
            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            info = Ref{BlasInt}()
            if eltype(A) <: Complex
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ref{BlasInt}, Ref{BlasInt}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ref{BlasInt}, Clong, Clong, Clong, Clong),
                       balanc, jobvl, jobvr, sense,
                       n, A, max(1,stride(A,2)), ws.W,
                       ws.VL, max(1,ldvl), ws.VR, max(1,ldvr),
                       ilo, ihi, ws.scale, abnrm,
                       ws.rconde, ws.rcondv, ws.work, length(ws.work),
                       ws.rwork, info, 1, 1, 1, 1)
                chklapackerror(info[])
                return A, ws.W, ws.VL, ws.VR, ilo[], ihi[], ws.scale, abnrm[], ws.rconde, ws.rcondv
            else
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                     Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                     Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                     Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                     Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                     Clong, Clong, Clong, Clong),
                     balanc, jobvl, jobvr, sense,
                     n, A, max(1, stride(A,2)), ws.W,
                     ws.rwork, ws.VL, max(1,ldvl), ws.VR,
                     max(1,ldvr), ilo, ihi, ws.scale,
                     abnrm, ws.rconde, ws.rcondv, ws.work,
                     length(ws.work), ws.iwork, info,
                     1, 1, 1, 1)
                chklapackerror(info[])
                return A, ws.W, ws.rwork, ws.VL, ws.VR, ilo[], ihi[], ws.scale, abnrm[], ws.rconde, ws.rcondv
            end
        end

    end
end

struct HermitianEigenWs{T, MT <: AbstractMatrix{T}, RT<:AbstractFloat}
    work::Vector{T}
    rwork::Vector{RT}
    iwork::Vector{BlasInt}
    w::Vector{RT}
    Z::MT
    isuppz::Vector{BlasInt}
end

function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, ws::HermitianEigenWs)
    summary(io, ws)
    println(io)
    for f in fieldnames(HermitianEigenWs)
        print(io, "$f: ")
        summary(io, getfield(ws, f))
        print(io, "\n")
    end
end

for (syevr, elty, relty) in ((:zheevr_,:ComplexF64,:Float64),
                             (:cheevr_,:ComplexF32,:Float32),
                             (:dsyevr_,:Float64, :Float64),
                             (:ssyevr_,:Float32, :Float32))
    @eval begin
        function HermitianEigenWs(A::AbstractMatrix{$elty}; vecs = true)
            chkstride1(A)
            n = checksquare(A)
            w = similar(A, $relty, n)
            if vecs
                ldz = n
                Z = similar(A, $elty, ldz, n)
            else
                ldz = 1
                Z = similar(A, $elty, ldz, 0)
            end
            isuppz = similar(A, BlasInt, 2*n)

            work   = Vector{$elty}(undef, 1)
            lwork  = BlasInt(-1)

            iwork  = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info   = Ref{BlasInt}()
            jobz = vecs ? 'V' : 'N'
            cmplx = eltype(A) <: Complex
            if cmplx
                rwork  = Vector{$relty}(undef, 1)
                lrwork = BlasInt(-1)
                ccall((@blasfunc($syevr), liblapack), Cvoid,
                        (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                        Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                        Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                        Clong, Clong, Clong),
                        jobz, 'A', 'U', n,
                        A, max(1, stride(A, 2)), 0, 0,
                        0, 0, 1e-6, Ref{BlasInt}(),
                        w, Z, ldz, isuppz,
                        work, lwork, rwork, lrwork,
                        iwork, liwork, info,
                        1, 1, 1)
                chklapackerror(info[])
                resize!(rwork, BlasInt(real(rwork[1])))
            else
                rwork  = Vector{$relty}(undef, 0)
                ccall((@blasfunc($syevr), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Ptr{BlasInt}, Clong, Clong, Clong),
                    jobz, 'A', 'U', n,
                    A, max(1, stride(A, 2)), 0, 0,
                    0, 0, 1e-6, Ref{BlasInt}(),
                    w, Z, ldz, isuppz,
                    work, lwork, iwork, liwork,
                    info, 1, 1, 1)
                chklapackerror(info[])
            end
            resize!(work, BlasInt(real(work[1])))
            resize!(iwork, BlasInt(real(iwork[1])))
            HermitianEigenWs(work, rwork, iwork, w, Z, isuppz)
        end

        function syevr!(jobz::AbstractChar, range::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty},
                        vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer, abstol::AbstractFloat, ws::HermitianEigenWs)
            chkstride1(A)
            n = checksquare(A)
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu=$iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            m = Ref{BlasInt}()
            
            if jobz == 'N'
                ldz = 1
            elseif jobz == 'V'
                ldz = n
            end
            
            info   = Ref{BlasInt}()
            if eltype(A) <: Complex
                ccall((@blasfunc($syevr), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ptr{BlasInt},
                       Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                       Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                       Clong, Clong, Clong),
                      jobz, range, uplo, n,
                      A, max(1,stride(A,2)), vl, vu,
                      il, iu, abstol, m,
                      ws.w, ws.Z, ldz, ws.isuppz,
                      ws.work, length(ws.work), ws.rwork, length(ws.rwork),
                      ws.iwork, length(ws.iwork), info,
                      1, 1, 1)
            else
                ccall((@blasfunc($syevr), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ref{$elty}, Ref{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Ref{BlasInt}, Clong, Clong, Clong),
                    jobz, range, uplo, n,
                    A, max(1,stride(A,2)), vl, vu,
                    il, iu, abstol, m,
                    ws.w, ws.Z, ldz, ws.isuppz,
                    ws.work, length(ws.work), ws.iwork, length(ws.iwork),
                    info, 1, 1, 1)
            end
            chklapackerror(info[])
            if range == 'A'
                return ws.w, ws.Z
            elseif range == 'I'
                return ws.w[1:iu-il+1], ws.Z[:,1:(jobz == 'V' ? iu-il+1 : 0)]
            else
                return ws.w[1:m[]], ws.Z[:,1:(jobz == 'V' ? m[] : 0)]
            end
        end            
    end
end


struct GeneralizedEigenWs
end
