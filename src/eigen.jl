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
struct EigenWs{T,MT<:AbstractMatrix{T},RT<:AbstractFloat} <: Workspace
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
        function EigenWs(A::AbstractMatrix{$elty}; lvecs = false, rvecs = true,
                         sense = false)
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

            VL    = zeros($elty, (lvecs ? n : 0, n))
            VR    = zeros($elty, (rvecs ? n : 0, n))
            cmplx = eltype(A) <: Complex
            if cmplx
                W     = zeros($elty, n)
                rwork = zeros($relty, 2n)
            else
                W     = zeros($elty, n)
                rwork = zeros($elty, n)
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
                      n, A, max(1, stride(A, 2)), W,
                      VL, n, VR, n,
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
                      n, A, max(1, stride(A, 2)), W,
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

        function geevx!(ws::EigenWs, balanc::AbstractChar, jobvl::AbstractChar,
                        jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix{$elty})
            n = checksquare(A)
            chkfinite(A) # balancing routines don't support NaNs and Infs
            if balanc ∉ ('N', 'P', 'S', 'B')
                throw(ArgumentError("balanc must be 'N', 'P', 'S', or 'B', but $balanc was passed"))
            end
            if sense ∉ ('N', 'E', 'V', 'B')
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
    geevx!(ws, balanc, jobvl, jobvr, sense, A) -> (A, ws.W, [ws.rwork,] ws.VL, ws.VR, ilo, ihi, ws.scale, abnrm, ws.rconde, ws.rcondv)

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
"""
geevx!(ws::EigenWs, balanc::AbstractChar, jobvl::AbstractChar,
       jobvr::AbstractChar, sense::AbstractChar, A::AbstractMatrix)

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
struct HermitianEigenWs{T,MT<:AbstractMatrix{T},RT<:AbstractFloat} <: Workspace
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
        function HermitianEigenWs(A::AbstractMatrix{$elty}; vecs = true)
            chkstride1(A)
            n = checksquare(A)
            w = zeros($relty, n)
            if vecs
                ldz = n
                Z = zeros($elty, ldz, n)
            else
                ldz = 1
                Z = zeros($elty, ldz, 0)
            end
            isuppz = zeros(BlasInt, 2 * n)

            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)

            iwork  = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info   = Ref{BlasInt}()
            jobz   = vecs ? 'V' : 'N'
            cmplx  = eltype(A) <: Complex
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
                rwork = Vector{$relty}(undef, 0)
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
            return HermitianEigenWs(work, rwork, iwork, w, Z, isuppz)
        end

        function syevr!(ws::HermitianEigenWs, jobz::AbstractChar, range::AbstractChar,
                        uplo::AbstractChar, A::AbstractMatrix{$elty},
                        vl::AbstractFloat, vu::AbstractFloat, il::Integer, iu::Integer,
                        abstol::AbstractFloat)
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
                return ws.w[1:iu-il+1], ws.Z[:, 1:(jobz == 'V' ? iu - il + 1 : 0)]
            else
                return ws.w[1:m[]], ws.Z[:, 1:(jobz == 'V' ? m[] : 0)]
            end
        end
    end
end

"""
    syevr!(ws, jobz, range, uplo, A, vl, vu, il, iu, abstol) -> (ws.W, ws.Z)

Finds the eigenvalues (`jobz = N`) or eigenvalues and eigenvectors
(`jobz = V`) of a symmetric matrix `A` using a preallocated [`HermitianEigenWs`](@ref).
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
       abstol::AbstractFloat)

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
struct GeneralizedEigenWs{T,MT<:AbstractMatrix{T},RT<:AbstractFloat} <: Workspace
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
        function GeneralizedEigenWs(A::AbstractMatrix{$elty}; lvecs = false, rvecs = false)
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            lda = max(1, stride(A, 2))
            ldb = lda
            αr = zeros($elty, n)
            cmplx = eltype(A) <: Complex
            αi = cmplx ? Vector{$relty}(undef, 8n) : zeros($relty, n)
            β = zeros($elty, n)

            jobvl = lvecs ? 'V' : 'N'
            jobvr = rvecs ? 'V' : 'N'
            ldvl = lvecs ? n : 1
            ldvr = rvecs ? n : 1
            vl = zeros($elty, lvecs ? n : 0, n)
            vr = zeros($elty, rvecs ? n : 0, n)

            work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()

            if cmplx
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty},
                       Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A,
                      lda, A, ldb, αr,
                      β, vl, ldvl, vr,
                      ldvr, work, lwork, αi,
                      info, 1, 1)
            else
                ccall((@blasfunc($ggev), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty},
                       Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                       Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                       Ptr{BlasInt}, Clong, Clong),
                      jobvl, jobvr, n, A,
                      lda, A, ldb, αr,
                      αi, β, vl, ldvl,
                      vr, ldvr, work, lwork,
                      info, 1, 1)
            end
            chklapackerror(info[])
            resize!(work, BlasInt(work[1]))
            return GeneralizedEigenWs(work, vl, vr, αr, αi, β)
        end

        function ggev!(ws::GeneralizedEigenWs, jobvl::AbstractChar, jobvr::AbstractChar,
                       A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty})
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            n, m = checksquare(A, B)
            if n != m
                throw(DimensionMismatch("A has dimensions $(size(A)), and B has dimensions $(size(B)), but A and B must have the same size"))
            end
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldvl = size(ws.vl, 1)
            ldvr = size(ws.vr, 1)
            @assert jobvl != 'V' || ldvl > 0 "Workspace was created without support for left eigenvectors,\nrecreate with EigenWs(A, lvecs = true)"
            @assert jobvr != 'V' || ldvr > 0 "Workspace was created without support for right eigenvectors,\nrecreate with EigenWs(A, rvecs = true)"
            rwork = ws.αi
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
    ggev!(ws, jobvl, jobvr, A, B) -> (ws.αr, [ws.αi,], ws.β, ws.vl, ws.vr)

Finds the generalized eigendecomposition of `A` and `B` usin a preallocated [`GeneralizedEigenWs`](@ref).
If `jobvl = N`, the left eigenvectors aren't computed. If `jobvr = N`, the right
eigenvectors aren't computed. If `jobvl = V` or `jobvr = V`, the
corresponding eigenvectors are computed. `ws.αi` is only returned in the `Real` case.
"""
ggev!(ws::GeneralizedEigenWs, jobvl::AbstractChar, jobvr::AbstractChar, A::AbstractMatrix,
      B::AbstractMatrix)
