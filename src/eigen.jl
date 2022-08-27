# Workspaces for Eigenvalue decomposition
using LinearAlgebra.LAPACK: chkfinite
import LinearAlgebra.LAPACK: geevx!, syevr!, ggev!

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

julia> ws = EigenWs(A, rvecs=true)
EigenWs{Float64, Matrix{Float64}, Float64}
  work: 260-element Vector{Float64}
  rwork: 2-element Vector{Float64}
  VL: 0×2 Matrix{Float64}
  VR: 2×2 Matrix{Float64}
  W: 2-element Vector{Float64}
  scale: 2-element Vector{Float64}
  iwork: 0-element Vector{Int64}
  rconde: 0-element Vector{Float64}
  rcondv: 0-element Vector{Float64}


julia> t = LAPACK.geevx!(ws, 'N', 'N', 'V', 'N', A);

julia> LinearAlgebra.Eigen(t[2], t[5])
Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
values:
2-element Vector{Float64}:
 -1.6695025194532018
  6.169502519453203
vectors:
2×2 Matrix{Float64}:
 -0.625424  -0.420019
  0.780285  -0.907515
```
"""
mutable struct EigenWs{T,MT<:AbstractMatrix{T},RT<:AbstractFloat} <: Workspace
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

for (geevx, elty, relty) in
    ((:dgeevx_, :Float64, :Float64),
     (:sgeevx_, :Float32, :Float32),
     (:zgeevx_, :ComplexF64, :Float64),
     (:cgeevx_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(ws::EigenWs, A::AbstractMatrix{$elty}; lvecs=false, rvecs=true, sense=false)
            
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

            ws.VL = zeros($elty, (lvecs ? n : 0, n))
            ws.VR = zeros($elty, (rvecs ? n : 0, n))
            cmplx = eltype(A) <: Complex
            resize!(ws.W, n) 
            resize!(ws.rwork, cmplx ? 2n : n)
                
            info  = Ref{BlasInt}()

            ilo = Ref{BlasInt}()
            ihi = Ref{BlasInt}()
            resize!(ws.scale, n)
            abnrm = Ref{$relty}()

            resize!(ws.rconde, sense ? n : 0)
            resize!(ws.rcondv, sense ? n : 0)

            resize!(ws.iwork, sense ? 2n - 1 : 0)

            if cmplx
                ccall((@blasfunc($geevx), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{UInt8},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Ptr{BlasInt}, Ptr{$relty}, Ptr{$relty},
                       Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$relty}, Ptr{BlasInt}, Clong, Clong, Clong, Clong),
                      'N', jobvl, jobvr, S,
                      n, A, max(1, stride(A, 2)), ws.W,
                      ws.VL, n, ws.VR, n,
                      ilo, ihi, ws.scale, abnrm,
                      ws.rconde, ws.rcondv, ws.work, -1,
                      ws.rwork, info, 1, 1, 1, 1)

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
                      n, A, max(1, stride(A, 2)), ws.W,
                      ws.rwork, ws.VL, n, ws.VR,
                      n, ilo, ihi, ws.scale,
                      abnrm, ws.rconde, ws.rcondv, ws.work,
                      -1, ws.iwork, info,
                      1, 1, 1, 1)
            end
            chklapackerror(info[])
            resize!(ws.work, BlasInt(real(ws.work[1])))
            return ws
        end
        EigenWs(A::AbstractMatrix{$elty}; kwargs...) =
            resize!(EigenWs(Vector{$elty}(undef, 1), $relty[], similar(A, 0, 0), similar(A, 0, 0), $elty[], $relty[], BlasInt[], $relty[], $relty[]), A; kwargs...)

        function geevx!(ws::EigenWs, balanc::AbstractChar, jobvl::AbstractChar,
                        jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty}; resize=true)
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            if balanc ∉ ('N', 'P', 'S', 'B')
                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
            end
            if sense ∉ ('N', 'E', 'V', 'B')
                throw(ArgumentError("sense must be 'N', 'E', 'V' or 'B', but $sense was passed"))
            end
            # work, rwork, VL, VR, W, scale, iwork, rconde, rcondv = ws
            if sense ∈ ('E', 'B')
                if jobvl != 'V' || jobvr != 'V'
                    throw(ArgumentError("If sense = $sense it is required that jobvl = 'V' (is $jobvl) and jobvr = 'V' (is $jobvr)."))
                elseif size(ws.iwork, 1) == 0
                    if resize
                        resize!(ws, A, sense=true)
                    else
                        throw(ArgumentError("Workspace was created without support for sense,\n use resize!(ws, A, sense=true)."))
                    end
                end
            end
            
            if jobvl == 'V' && size(ws.VL, 1) == 0
                if resize
                    resize!(ws, A, lvecs = true, rvecs = size(ws.VR, 1) != 0, sense=size(ws.iwork, 1) != 0)
                else
                    throw(ArgumentError("Workspace was created without support for left eigenvectors,\n use resize!(ws, A, lvecs=true)."))
                end
            end
            if jobvr == 'V' && size(ws.VR, 1) == 0
                if resize
                    resize!(ws, A, rvecs = true, lvecs = size(ws.VL, 1) != 0, sense=size(ws.iwork, 1) != 0)
                else
                    throw(ArgumentError("Workspace was created without support for right eigenvectors,\nor use resize!(ws, A, rvecs=true)."))
                end
            end

            ldvl = size(ws.VL, 1)
            ldvr = size(ws.VR, 1)
            nws = length(ws.W)
            if n != nws
                if resize
                    resize!(ws, A, rvecs = ldvr != 0, lvecs = ldvl != 0, sense=size(ws.iwork, 1) != 0)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            
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
                      n, A, max(1, stride(A, 2)), ws.W,
                      ws.VL, max(1, ldvl), ws.VR, max(1, ldvr),
                      ilo, ihi, ws.scale, abnrm,
                      ws.rconde, ws.rcondv, ws.work, length(ws.work),
                      ws.rwork, info, 1, 1, 1, 1)
                chklapackerror(info[])
                return A, ws.W, ws.VL, ws.VR, ilo[], ihi[], ws.scale, abnrm[], ws.rconde,
                       ws.rcondv
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
                      n, A, max(1, stride(A, 2)), ws.W,
                      ws.rwork, ws.VL, max(1, ldvl), ws.VR,
                      max(1, ldvr), ilo, ihi, ws.scale,
                      abnrm, ws.rconde, ws.rcondv, ws.work,
                      length(ws.work), ws.iwork, info,
                      1, 1, 1, 1)
                chklapackerror(info[])
                return A, ws.W, ws.rwork, ws.VL, ws.VR, ilo[], ihi[], ws.scale, abnrm[],
                       ws.rconde, ws.rcondv
            end
        end
    end
end

"""
    geevx!(ws, balanc, jobvl, jobvr, sense, A; resize=true) -> (A, ws.W, [ws.rwork,] ws.VL, ws.VR, ilo, ihi, ws.scale, abnrm, ws.rconde, ws.rcondv)

Finds the eigensystem of `A` with matrix balancing using a preallocated [`EigenWs`](@ref).
If `jobvl = N`, the left eigenvectors of `A` aren't computed. If `jobvr = N`, the right
eigenvectors of `A` aren't computed. If `jobvl = V` or `jobvr = V`, the
corresponding eigenvectors are computed. If `balanc = N`, no balancing is
performed. If `balanc = P`, `A` is permuted but not scaled. If
`balanc = S`, `A` is scaled but not permuted. If `balanc = B`, `A` is
permuted and scaled. If `sense = N`, no reciprocal condition numbers are
computed. If `sense = E`, reciprocal condition numbers are computed for
the eigenvalues only. If `sense = V`, reciprocal condition numbers are
computed for the right eigenvectors only. If `sense = B`, reciprocal
condition numbers are computed for the right eigenvectors and the
eigenvectors. If `sense = E,B`, the right and left eigenvectors must be
computed. `ws.rwork` is only returned in the `Real` case.
If `ws` does not have the appropriate size for `A` and the work to be done,
if `resize=true`, it will be automatically resized accordingly. 
"""
geevx!(ws::EigenWs, balanc::AbstractChar, jobvl::AbstractChar,
       jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix; resize=true)

"""
    HermitianEigenWs

Workspace to be used with Hermitian diagonalization using the [`LAPACK.syevr!`](@ref) function.
Supports both `Real` and `Complex` Hermitian matrices.
# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = HermitianEigenWs(A, vecs=true)
HermitianEigenWs{Float64, Matrix{Float64}, Float64}
  work: 66-element Vector{Float64}
  rwork: 0-element Vector{Float64}
  iwork: 20-element Vector{Int64}
  w: 2-element Vector{Float64}
  Z: 2×2 Matrix{Float64}
  isuppz: 4-element Vector{Int64}


julia> LinearAlgebra.Eigen(LAPACK.syevr!(ws, 'V', 'A', 'U', A, 0.0, 0.0, 0, 0, 1e-6)...)
Eigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
values:
2-element Vector{Float64}:
 -0.2783393759541063
  4.778339375954106
vectors:
2×2 Matrix{Float64}:
 -0.841217  0.540698
  0.540698  0.841217
```
"""
mutable struct HermitianEigenWs{T,MT<:AbstractMatrix{T},RT<:AbstractFloat} <: Workspace
    work::Vector{T}
    rwork::Vector{RT}
    iwork::Vector{BlasInt}
    w::Vector{RT}
    Z::MT
    isuppz::Vector{BlasInt}
end

for (syevr, elty, relty) in ((:zheevr_, :ComplexF64, :Float64),
                             (:cheevr_, :ComplexF32, :Float32),
                             (:dsyevr_, :Float64, :Float64),
                             (:ssyevr_, :Float32, :Float32))
    @eval begin
        function Base.resize!(ws::HermitianEigenWs, A::AbstractMatrix{$elty}; vecs=false, work=true)
            chkstride1(A)
            n = checksquare(A)
            resize!(ws.w, n)
            if vecs
                ldz = n
                ws.Z = similar(A, ldz, n)
            else
                ldz = 1
                ws.Z = similar(A, 0, 0)
            end
            resize!(ws.isuppz, 2n)

            if work
                info   = Ref{BlasInt}()
                jobz   = vecs ? 'V' : 'N'
                cmplx  = eltype(A) <: Complex
                if cmplx
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
                          ws.w, ws.Z, ldz, ws.isuppz,
                          ws.work, -1, ws.rwork, -1,
                          ws.iwork, -1, info,
                          1, 1, 1)
                    chklapackerror(info[])
                    resize!(ws.rwork, BlasInt(real(ws.rwork[1])))
                else
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
                          ws.w, ws.Z, ldz, ws.isuppz,
                          ws.work, -1, ws.iwork, -1,
                          info, 1, 1, 1)
                    chklapackerror(info[])
                end
                resize!(ws.work, BlasInt(real(ws.work[1])))
                resize!(ws.iwork, BlasInt(real(ws.iwork[1])))
            end
            return ws
        end
        function HermitianEigenWs(A::AbstractMatrix{$elty}; kwargs...)
            return resize!(HermitianEigenWs(Vector{$elty}(undef, 1), Vector{$relty}(undef, eltype(A) <: Complex ? 1 : 0), Vector{BlasInt}(undef, 1), $relty[], similar(A, 0, 0), BlasInt[]), A; kwargs...)
        end

        function syevr!(ws::HermitianEigenWs, jobz::AbstractChar, range::AbstractChar,
                        uplo::AbstractChar, A::AbstractMatrix{$elty},
                        vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer,
                        abstol::AbstractFloat; resize=true)
                        
            if range == 'I' && !(1 <= il <= iu <= n)
                throw(ArgumentError("illegal choice of eigenvalue indices (il = $il, iu=$iu), which must be between 1 and n = $n"))
            end
            if range == 'V' && vl >= vu
                throw(ArgumentError("lower boundary, $vl, must be less than upper boundary, $vu"))
            end
            
            chkstride1(A)
            n = checksquare(A)
            nws = length(ws.w)
            if nws != n
                if resize
                    resize!(ws, A, vecs = size(ws.Z, 1) > 1 || jobz == 'V', work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end

            # If WS was created without support for vectors
            if jobz == 'N'
                ldz = 1
            elseif jobz == 'V'
                ldz = n
                nws = size(ws.Z, 1)
                if nws != ldz
                    if resize
                        ws.Z = similar(ws.Z, ldz, ldz)
                    else
                        throw(ArgumentError("Workspace does not support eigenvectors.\nUse resize!(ws, A, vecs=true)."))
                    end
                end
            end
                   
            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
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
                      A, max(1, stride(A, 2)), vl, vu,
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
                      A, max(1, stride(A, 2)), vl, vu,
                      il, iu, abstol, m,
                      ws.w, ws.Z, ldz, ws.isuppz,
                      ws.work, length(ws.work), ws.iwork, length(ws.iwork),
                      info, 1, 1, 1)
            end
            chklapackerror(info[])
            if range == 'A'
                return ws.w, ws.Z
            elseif range == 'I'
                return ws.w[1:iu-il+1], ws.Z[1:ldz, 1:(jobz == 'V' ? iu - il + 1 : 0)]
            else
                return ws.w[1:m[]], ws.Z[1:ldz, 1:(jobz == 'V' ? m[] : 0)]
            end
        end
    end
end

"""
    syevr!(ws, jobz, range, uplo, A, vl, vu, il, iu, abstol; resize=true) -> (ws.W, ws.Z)

Finds the eigenvalues (`jobz = N`) or eigenvalues and eigenvectors
(`jobz = V`) of a symmetric matrix `A` using a preallocated [`HermitianEigenWs`](@ref).
If the workspace is not appropriate for `A` and `resize==true` it will be automatically
resized.
If `uplo = U`, the upper triangle of `A` is used. If `uplo = L`, the lower triangle of `A` is used.
If `range = A`, all the eigenvalues are found. If `range = V`, the
eigenvalues in the half-open interval `(vl, vu]` are found.
If `range = I`, the eigenvalues with indices between `il` and `iu` are
found. `abstol` can be set as a tolerance for convergence.

The eigenvalues are returned as `ws.W` and the eigenvectors in `ws.Z`.
"""
syevr!(ws::HermitianEigenWs, jobz::AbstractChar, range::AbstractChar,
       uplo::AbstractChar, A::AbstractMatrix,
       vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer,
       abstol::AbstractFloat; resize=true)

"""
    GeneralizedEigenWs

Workspace that can be used for [`LinearAlgebra.GeneralizedEigen`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedEigen)
factorization using [`LAPACK.ggev!`](@ref).
Supports `Real` and `Complex` matrices.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> B = [8.2 1.7
            5.9 2.1]
2×2 Matrix{Float64}:
 8.2  1.7
 5.9  2.1

julia> ws = GeneralizedEigenWs(A, rvecs=true)
GeneralizedEigenWs{Float64, Matrix{Float64}, Float64}
  work: 78-element Vector{Float64}
  vl: 0×2 Matrix{Float64}
  vr: 2×2 Matrix{Float64}
  αr: 2-element Vector{Float64}
  αi: 2-element Vector{Float64}
  β: 2-element Vector{Float64}


julia> αr, αi, β, _, vr = LAPACK.ggev!(ws, 'N', 'V', A, B);

julia> LinearAlgebra.GeneralizedEigen(αr ./ β, vr)
GeneralizedEigen{Float64, Float64, Matrix{Float64}, Vector{Float64}}
values:
2-element Vector{Float64}:
 -0.8754932558185097
  1.6362721153456299
vectors:
2×2 Matrix{Float64}:
 -0.452121  -0.0394242
  1.0        1.0
```
"""
mutable struct GeneralizedEigenWs{T,MT<:AbstractMatrix{T},RT<:AbstractFloat} <: Workspace
    work::Vector{T}
    vl::MT
    vr::MT
    αr::Vector{T}
    αi::Vector{RT}
    β::Vector{T}
end

for (ggev, elty, relty) in
    ((:dggev_, :Float64, :Float64),
     (:sggev_, :Float32, :Float32),
     (:zggev_, :ComplexF64, :Float64),
     (:cggev_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(ws::GeneralizedEigenWs, A::AbstractMatrix{$elty}; lvecs=false,rvecs=false,work=true)
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            ldb = lda
            resize!(ws.αr, n)
            cmplx = eltype(A) <: Complex
            if cmplx && work
                resize!(ws.αi, 8n)
            else
                resize!(ws.αi, n)
            end
            resize!(ws.β, n)

            jobvl = lvecs ? 'V' : 'N'
            jobvr = rvecs ? 'V' : 'N'
            ldvl = lvecs ? n : 1
            ldvr = rvecs ? n : 1
            ws.vl = zeros($elty, lvecs ? n : 0, n)
            ws.vr = zeros($elty, rvecs ? n : 0, n)
            if work
                info = Ref{BlasInt}()
                if cmplx
                    ccall((@blasfunc($ggev), liblapack), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                           Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr, n, A,
                          lda, A, ldb, ws.αr,
                          ws.β, ws.vl, ldvl, ws.vr,
                          ldvr, ws.work, -1, ws.αi,
                          info, 1, 1)
                else
                    ccall((@blasfunc($ggev), liblapack), Cvoid,
                          (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                           Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                           Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                           Ptr{BlasInt}, Clong, Clong),
                          jobvl, jobvr, n, A,
                          lda, A, ldb, ws.αr,
                          ws.αi, ws.β, ws.vl, ldvl,
                          ws.vr, ldvr, ws.work, -1,
                          info, 1, 1)
                end
                chklapackerror(info[])
                resize!(ws.work, BlasInt(ws.work[1]))
            end
            return ws
        end
        function GeneralizedEigenWs(A::AbstractMatrix{$elty}; kwargs...)
            return resize!(GeneralizedEigenWs(Vector{$elty}(undef, 1), similar(A, 0,0),similar(A,0,0), $elty[], $relty[], $elty[]), A; kwargs...)
        end

        function ggev!(ws::GeneralizedEigenWs, jobvl::AbstractChar, jobvr::AbstractChar,
                       A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}; resize=true)
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            n = checksquare(A)
            m = checksquare(B)
            if n != m
                throw(DimensionMismatch("A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))

            nws = length(ws.β)
            if nws != n
                if resize
                    resize!(ws, A, lvecs = size(ws.vl, 1) > 0 || jobvl == 'V', rvecs = size(ws.vr, 1) > 0 || jobvl == 'V', work=n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            
            ldvl = size(ws.vl, 1)
            ldvr = size(ws.vr, 1)

            for (job, v, str1, str2) in ((jobvl, :vl, "left", "lvecs"),(jobvr, :vr, "right", "rvecs"))
                ldv = size(getfield(ws, v), 1) 
                if job == 'V'
                    if ldv != n
                        if resize
                            setfield!(ws, v, similar(A, n, n))
                        else
                            throw(ArgumentError("Workspace was created without support for $str1 eigenvectors or too small,\n use resize!(ws, A, $str2=true)."))
                        end
                    end
                end
            end

            ldvl = size(ws.vl, 1)
            ldvr = size(ws.vr, 1)
                   
            info = Ref{BlasInt}()
            if eltype(A) <: Complex
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                       Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A,
                      lda, B, ldb, ws.αr,
                      ws.β, ws.vl, max(ldvl, 1), ws.vr,
                      max(ldvl, 1), ws.work, length(ws.work), ws.αi,
                      info, 1, 1)
                chklapackerror(info[])
                return ws.αr, ws.β, ws.vl, ws.vr
                    
            else
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A,
                      lda, B, ldb, ws.αr,
                      ws.αi, ws.β, ws.vl, max(ldvl, 1),
                      ws.vr, max(ldvr, 1), ws.work, length(ws.work),
                      info, 1, 1)
                chklapackerror(info[])
                return ws.αr, ws.αi, ws.β, ws.vl, ws.vr
            end
        end
    end
end

"""
    ggev!(ws, jobvl, jobvr, A, B; resize=true) -> (ws.αr, [ws.αi,], ws.β, ws.vl, ws.vr)

Finds the generalized eigendecomposition of `A` and `B` usin a preallocated [`GeneralizedEigenWs`](@ref).
If the workspace is not appropriately sized and `resize == true`, it will automatically be
resized.
If `jobvl = N`, the left eigenvectors aren't computed. If `jobvr = N`, the right
eigenvectors aren't computed. If `jobvl = V` or `jobvr = V`, the
corresponding eigenvectors are computed. `ws.αi` is only returned in the `Real` case.
"""
ggev!(ws::GeneralizedEigenWs, jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix,
      B::AbstractMatrix; resize=true)
