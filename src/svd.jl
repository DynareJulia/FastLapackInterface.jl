# Workspaces for Eigenvalue decomposition
using LinearAlgebra.LAPACK: chkfinite, chklapackerror
if VERSION >= v"1.11"
    using LinearAlgebra.LAPACK: @chkvalidparam
else
    macro chkvalidparam(position::Int, param, validvalues)
        :(chkvalidparam($position, $(string(param)), $(esc(param)), $validvalues))
    end
    function chkvalidparam(position::Int, var::String, val, validvals)
        # mimic `repr` for chars without explicitly calling it
        # This is because `repr` introduces dynamic dispatch
        _repr(c::AbstractChar) = "'$c'"
        _repr(c) = c
        if val ∉ validvals
            throw(ArgumentError(
                lazy"argument #$position: $var must be one of $validvals, but $(_repr(val)) was passed"))
        end
        return val
    end
end
import LinearAlgebra.LAPACK: gesdd!, gesvd!, ggsvd3!

"""
    SVDsddWs

creates a workspace for [`LinearAlgebra.SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SVD)
factorization using the [`LAPACK.gesdd!`](@ref) function.
"""
mutable struct SVDsddWs{T, MT <: AbstractMatrix{T}, RT <: AbstractFloat} <: Workspace
    U::MT
    VT::MT
    work::Vector{T}
    S::Vector{RT}
    rwork::Vector{RT} # Can be rwork if T <: Complex or WI if T <: Float64
    iwork::Vector{BlasInt}
end

# (GE) general matrices ingular value decomposition
for (gesdd, elty, relty) in ((:dgesdd_, :Float64, :Float64),
    (:sgesdd_, :Float32, :Float32),
    (:zgesdd_, :ComplexF64, :Float64),
    (:cgesdd_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(
            ws::SVDsddWs, A::AbstractMatrix{$elty}; job::AbstractChar = 'A')
            require_one_based_indexing(A)
            chkstride1(A)
            if VERSION >= v"1.11"
                @chkvalidparam 2 job ('A', 'S', 'O', 'N')
            end
            m, n = size(A)
            minmn = min(m, n)
            if job == 'A'
                ws.U = similar(A, $elty, (m, m))
                ws.VT = similar(A, $elty, (n, n))
            elseif job == 'S'
                ws.U = similar(A, $elty, (m, minmn))
                ws.VT = similar(A, $elty, (minmn, n))
            elseif job == 'O'
                ws.U = similar(A, $elty, (m, m >= n ? 0 : m))
                ws.VT = similar(A, $elty, (n, m >= n ? n : 0))
            else
                ws.U = similar(A, $elty, (m, 0))
                ws.VT = similar(A, $elty, (n, 0))
            end
            ws.work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            ws.S = similar(A, $relty, minmn)
            cmplx = eltype(A) <: Complex
            if cmplx
                ws.rwork = Vector{$relty}(undef,
                    job == 'N' ? 7 * minmn :
                    minmn * max(5 * minmn + 7, 2 * max(m, n) + 2 * minmn + 1))
            end
            ws.iwork = Vector{BlasInt}(undef, 8 * minmn)
            info = Ref{BlasInt}()
            if cmplx
                ccall((@blasfunc($gesdd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{BlasInt}, Ref{BlasInt}, Clong),
                    job, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, lwork, ws.rwork, ws.iwork, info, 1)
            else
                ccall((@blasfunc($gesdd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{BlasInt}, Ref{BlasInt}, Clong),
                    job, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, lwork, ws.iwork, info, 1)
            end
            chklapackerror(info[])
            # Work around issue with truncated Float32 representation of lwork in
            # sgesdd by using nextfloat. See
            # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
            # and
            # https://github.com/scipy/scipy/issues/5401
            lwork = round(BlasInt, nextfloat(real(ws.work[1])))
            resize!(ws.work, lwork)
            return ws
        end
        
        SVDsddWs(A::AbstractMatrix{$elty}; kwargs...) = Base.resize!(
            SVDsddWs(similar(A, 0, 0), similar(A, 0, 0), Vector{$elty}(undef, 1),
                Vector{$relty}(undef, 0), Vector{$relty}(undef, 0), Vector{BlasInt}(
                    undef, 0)),
            A;
            kwargs...)
        #    SUBROUTINE DGESDD( JOBZ, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK,
        #                   LWORK, IWORK, INFO )
        #*     .. Scalar Arguments ..
        #      CHARACTER          JOBZ
        #      INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
        #*     ..
        #*     .. Array Arguments ..
        #      INTEGER            IWORK( * )
        #      DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ),
        #                        VT( LDVT, * ), WORK( * )
        function gesdd!(ws::SVDsddWs, job::AbstractChar, A::AbstractMatrix{$elty}; resize = true)
            require_one_based_indexing(A)
            chkstride1(A)
            if VERSION >= v"1.11"
                @chkvalidparam 1 job ('A', 'S', 'O', 'N')
            end
            m, n = size(A)
            minmn = min(m, n)
            if job == 'A'
                if size(ws.U) != (m, n) || size(ws.VT) != (n, n)
                    if resize
                        resize!(ws, A, job = job)
                    else
                        throw(ArgumentError("Workspace has wrong size, use resize!(ws, A)."))
                    end
                end
            elseif job == 'S'
                if size(ws.U) != (m, minmn) || size(ws.VT) != (minmn, n)
                    if resize
                        resize!(ws, A, job = job)
                    else
                        throw(ArgumentError("Workspace has wrong size, use resize!(ws, A, job = 'S')."))
                    end
                end
            elseif job == 'O'
                if m >= n
                    if size(ws.U, 1) != m || size(ws.VT) != (n, n)
                        if resize
                            resize!(ws, A, job = job)
                        else
                            throw(ArgumentError("Workspace has wrong size, use resize!(ws, A, job = 'O')."))
                        end
                    end
                else
                    if size(ws.U) != (m, m) || size(ws.VT, 1) != n
                        if resize
                            resize!(ws, A, job = job)
                        else
                            throw(ArgumentError("Workspace has wrong size, use resize!(ws, A, job = 'O')."))
                        end
                    end
                end
            end
            cmplx = eltype(A) <: Complex
            info = Ref{BlasInt}()
            if cmplx
                ccall((@blasfunc($gesdd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{BlasInt}, Ref{BlasInt}, Clong),
                    job, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, length(ws.work), ws.rwork, ws.iwork, info, 1)
            else
                ccall((@blasfunc($gesdd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{BlasInt}, Ref{BlasInt}, Clong),
                      job, m, n, A,
                      max(1, stride(A, 2)), ws.S, ws.U, max(1, stride(ws.U, 2)),
                      ws.VT, max(1, stride(ws.VT, 2)), ws.work, length(ws.work),
                      ws.iwork, info, 1)
            end
            chklapackerror(info[])
            if job == 'O'
                if m >= n
                    return (A, ws.S, ws.VT)
                else
                    return (ws.U, ws.S, A)
                end
            end
            return (ws.U, ws.S, ws.VT)
        end
    end
end


"""
    SVDsddWs(A; job = 'A')

creates a workspace for [`LinearAlgebra.SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SVD)
factorization using the [`LAPACK.gesdd!`](@ref) function.

# Arguments
- `A::AbstractMatrix`: the matrix to be decomposed. Only the shape of the matrix matters for building the workspace. The actual ellements value does not.
- `job::AbstractChar`: one of `A`, `N`, `O` or `S`. Default = 'A'. See [gesdd!](@ref).

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = FastLapackInterface.SVDsddWs(A, job = 'A')
SVDsddWs{Float64, Matrix{Float64}, Float64}
  U: 2×2 Matrix{Float64}
  VT: 2×2 Matrix{Float64}
  work: 134-element Vector{Float64}
  S: 2-element Vector{Float64}
  rwork: 0-element Vector{Float64}
  iwork: 16-element Vector{Int64}

julia> t = FastLapackInterface.gesdd!(ws, 'A', A);

julia> LinearAlgebra.SVD(t[1], t[2], t[3])
SVD{Float64, Float64, Matrix{Float64}, Vector{Float64}}
U factor:
2×2 Matrix{Float64}:
 -0.302437  -0.953169
 -0.953169   0.302437
singular values:
2-element Vector{Float64}:
 7.355199814161389
 1.400369841777625
Vt factor:
2×2 Matrix{Float64}:
 -0.852808  -0.522224
  0.522224  -0.852808
```
"""
SVDsddWs(::AbstractMatrix; job::AbstractChar)

"""
    gesdd!(ws, job, A; resize = true) -> (U, S, VT)

Finds the singular value decomposition of `A`, `A = U * S * V'`,
using a divide and conquer approachusing a preallocated [`SVDsddWs`](@ref).
If `job : A`, all the columns of `U` and
the rows of `V'` are computed. If `job = N`, no columns of `U` or rows of `V'`
are computed. If `job = O`, `A` is overwritten with the columns of (thin) `U`
and the rows of (thin) `V'`. If `job = S`, the columns of (thin) `U` and the
rows of (thin) `V'` are computed and returned separately. The value of `job` must be the same in `SVDsddWs` and in 'gesdd!`.
If `ws` does not have the appropriate size for `A` and the work to be done,
if `resize=true`, it will be automatically resized accordingly. 
"""
gesdd!(ws::SVDsddWs, job::AbstractChar, A::AbstractMatrix)

"""
    SVDsvdWs

Workspace for [`LinearAlgebra.SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SVD)
factorization using the [`LAPACK.gesvd!`](@ref) function.
"""
mutable struct SVDsvdWs{T, MT <: AbstractMatrix{T}, RT <: AbstractFloat} <: Workspace
    U::MT
    VT::MT
    work::Vector{T}
    S::Vector{RT}
    rwork::Vector{RT} # Can be rwork if T <: Complex or WI if T <: Float64
end

# (GE) general matrices ingular value decomposition
for (gesvd, elty, relty) in ((:dgesvd_, :Float64, :Float64),
    (:sgesvd_, :Float32, :Float32),
    (:zgesvd_, :ComplexF64, :Float64),
    (:cgesvd_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(ws::SVDsvdWs, A::AbstractMatrix{$elty};
                jobu::AbstractChar = 'A', jobvt::AbstractChar = 'A')
            require_one_based_indexing(A)
            chkstride1(A)
            if VERSION >= v"1.11"
                @chkvalidparam 2 jobu ('A', 'S', 'O', 'N')
                @chkvalidparam 3 jobvt ('A', 'S', 'O', 'N')
            end
            m, n = size(A)
            minmn = min(m, n)
            (jobu == jobvt == 'O') &&
                throw(ArgumentError("jobu and jobvt cannot both be O"))
            ws.S = similar(A, $relty, minmn)
            ws.U = similar(
                A, $elty, jobu == 'A' ? (m, m) : (jobu == 'S' ? (m, minmn) : (m, 0)))
            ws.VT = similar(
                A, $elty, jobvt == 'A' ? (n, n) : (jobvt == 'S' ? (minmn, n) : (n, 0)))
            cmplx = eltype(A) <: Complex
            if cmplx
                ws.rwork = Vector{$relty}(undef, 5minmn)
            end
            lwork = BlasInt(-1)
            info = Ref{BlasInt}()
            if cmplx
                ccall((@blasfunc($gesvd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt}, Clong, Clong),
                    jobu, jobvt, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, lwork, ws.rwork, info, 1, 1)
            else
                ccall((@blasfunc($gesvd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                    jobu, jobvt, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, lwork, info, 1, 1)
            end
            chklapackerror(info[])
            # Work around issue with truncated Float32 representation of lwork in
            # sgesdd by using nextfloat. See
            # http://icl.cs.utk.edu/lapack-forum/viewtopic.php?f=13&t=4587&p=11036&hilit=sgesdd#p11036
            # and
            # https://github.com/scipy/scipy/issues/5401
            lwork = round(BlasInt, nextfloat(real(ws.work[1])))
            Base.resize!(ws.work, lwork)
            return ws
        end
        SVDsvdWs(A::AbstractMatrix{$elty}; kwargs...) = Base.resize!(
            SVDsvdWs(similar(A, 0, 0), similar(A, 0, 0), Vector{$elty}(undef, 1),
                Vector{$relty}(undef, 0), Vector{$relty}(undef, 0)),
            A;
            kwargs...)

        # SUBROUTINE DGESVD( JOBU, JOBVT, M, N, A, LDA, S, U, LDU, VT, LDVT, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBU, JOBVT
        #       INTEGER            INFO, LDA, LDU, LDVT, LWORK, M, N
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), S( * ), U( LDU, * ),
        #                          VT( LDVT, * ), WORK( * )
        function gesvd!(ws::SVDsvdWs, jobu::AbstractChar,
                jobvt::AbstractChar, A::AbstractMatrix{$elty}; resize = true)
            require_one_based_indexing(A)
            chkstride1(A)
            if VERSION >= v"1.11"
                @chkvalidparam 2 jobu ('A', 'S', 'O', 'N')
                @chkvalidparam 3 jobvt ('A', 'S', 'O', 'N')
            end
            (jobu == jobvt == 'O') &&
                throw(ArgumentError("jobu and jobvt cannot both be O"))
            m, n = size(A)
            minmn = min(m, n)
            if jobu == 'A' && size(ws.U) != (m, n)
                if resize
                    resize!(ws, A, jobu = jobu, jobvt = jobvt)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            elseif jobu == 'S' && size(ws.U) != (m, minmn)
                if resize
                    resize!(ws, A, jobu = jobu, jobvt = jobvt)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            elseif jobu == 'O'
                if m >= n
                    if size(ws.U, 1) != m
                        if resize
                            resize!(ws, A, jobu = jobu, jobvt = jobvt)
                        else
                            throw(ArgumentError("Workspace has wrong size, use resize!()."))
                        end
                    end
                else
                    if size(ws.U) != (m, m)
                        if resize
                            resize!(ws, A, jobu = jobu, jobvt = jobvt)
                        else
                            throw(ArgumentError("Workspace has wrong size, use resize!()."))
                        end
                    end
                end
            end
            if jobvt == 'A' && size(ws.VT) != (n, n)
                if resize
                    resize!(ws, A, jobu = jobu, jobvt = jobvt)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            elseif jobvt == 'S' && size(ws.VT) != (minmn, n)
                if resize
                    resize!(ws, A, jobu = jobu, jobvt = jobvt)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            elseif jobvt == 'O'
                if m >= n
                    if size(ws.VT) != (n, n)
                        if resize
                            resize!(ws, A, jobu = jobu, jobvt = jobvt)
                        else
                            throw(ArgumentError("Workspace has wrong size, use resize!()."))
                        end
                    end
                else
                    if size(ws.VT, 1) != n
                        if resize
                            resize!(ws, A, jobu = jobu, jobvt = jobvt)
                        else
                            throw(ArgumentError("Workspace has wrong size, use resize!()."))
                        end
                    end
                end
            end
            cmplx = eltype(A) <: Complex
            info = Ref{BlasInt}()
            if cmplx
                ccall((@blasfunc($gesvd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt}, Clong, Clong),
                    jobu, jobvt, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, length(ws.work), ws.rwork, info, 1, 1)
            else
                ccall((@blasfunc($gesvd), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Clong, Clong),
                    jobu, jobvt, m, n, A, max(1, stride(A, 2)), ws.S, ws.U,
                    max(1, stride(ws.U, 2)), ws.VT, max(1, stride(ws.VT, 2)),
                    ws.work, length(ws.work), info, 1, 1)
            end
            chklapackerror(info[])
            if jobu == 'O'
                return (A, ws.S, ws.VT)
            elseif jobvt == 'O'
                return (ws.U, ws.S, A)
            else
                return (ws.U, ws.S, ws.VT)
            end
        end
    end
end

"""
    SVDsvdWs(A; jobu = 'A', jobvt = 'A') 

creates a workspace for [`LinearAlgebra.SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SVD)
factorization using the [`LAPACK.gesvd!`](@ref) function.

# Arguments
- `A::AbstractMatrix`: the matrix to be decomposed. Only the shape of the matrix matters for building the workspace. The actual ellements value does not.
- `jobu::AbstractChar`: one of `A`, `N`, `O` or `S`. Default = 'A'. See [gesvd!](@ref).
- `jobvt::AbstractChar`: one of `A`, `N`, `O` or `S`. Default = 'A'. See [gesvd!](@ref).

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = SVDsvdWs(A, jobu = 'A', jobvt = 'A')
SVDsvdWs{Float64, Matrix{Float64}, Float64}
  U: 2×2 Matrix{Float64}
  VT: 2×2 Matrix{Float64}
  work: 134-element Vector{Float64}
  S: 2-element Vector{Float64}
  rwork: 0-element Vector{Float64}

julia> t = FastLapackInterface.gesvd!(ws, 'A', 'A', A);

julia> SVD(t[1], t[2], t[3])
SVD{Float64, Float64, Matrix{Float64}, Vector{Float64}}
U factor:
2×2 Matrix{Float64}:
 -0.302437  -0.953169
 -0.953169   0.302437
singular values:
2-element Vector{Float64}:
 7.355199814161389
 1.400369841777625
Vt factor:
2×2 Matrix{Float64}:
 -0.852808  -0.522224
  0.522224  -0.852808
```
"""
SVDsvdWs(A::AbstractMatrix; jobu = 'A', jobvt = 'A')

"""
    gesvd!(ws, jobu, jobvt, A; resize = true) -> (U, S, VT)

Finds the singular value decomposition of `A`, `A = U * S * V'`
using a preallocated [`SVDsvdWs`](@ref).
If `jobu = A`, all the columns of `U` are computed. If `jobvt = A` all the rows
of `V'` are computed. If `jobu = N`, no columns of `U` are computed. If
`jobvt = N` no rows of `V'` are computed. If `jobu = O`, `A` is overwritten with
the columns of (thin) `U`. If `jobvt = O`, `A` is overwritten with the rows
of (thin) `V'`. If `jobu = S`, the columns of (thin) `U` are computed
and returned separately. If `jobvt = S` the rows of (thin) `V'` are
computed and returned separately. `jobu` and `jobvt` can't both be `O`. The value of `jobu` and `jobvt` must be the same in `SVDsvdWs` and in `gesvd!`

Returns `U`, `S`, and `Vt`, where `S` are the singular values of `A`.
If `ws` does not have the appropriate size for `A` and the work to be done,
if `resize=true`, it will be automatically resized accordingly. 
"""
gesvd!(ws::SVDsvdWs, jobu::AbstractChar, jobvt::AbstractChar, A::AbstractMatrix)

"""
    GeneralizedSVDWs

Workspace for [`LinearAlgebra.SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedSVD)
factorization using the [`LAPACK.ggsvd3!`](@ref) function.
"""
mutable struct GeneralizedSVDWs{T, MT <: AbstractMatrix{T}, RT <: AbstractFloat} <:
               Workspace
    alpha::Vector{RT}
    beta::Vector{RT}
    U::MT
    V::MT
    Q::MT
    work::Vector{T}
    rwork::Vector{RT} # Can be rwork if T <: Complex or WI if T <: Float64
    iwork::Vector{BlasInt}
    # add R
end

# (GE) general matrices generalized singular value decomposition
for (ggsvd3, elty, relty) in ((:dggsvd3_, :Float64, :Float64),
    (:sggsvd3_, :Float32, :Float32),
    (:zggsvd3_, :ComplexF64, :Float64),
    (:cggsvd3_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(ws::GeneralizedSVDWs, A::AbstractMatrix{$elty},
                B::AbstractMatrix{$elty}; jobu::AbstractChar = 'U',
                jobv::AbstractChar = 'V', jobq::AbstractChar = 'Q')
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            if VERSION >= v"1.11"
                @chkvalidparam 1 jobu ('U', 'N')
                @chkvalidparam 2 jobv ('V', 'N')
                @chkvalidparam 3 jobq ('Q', 'N')
            end
            m, n = size(A)
            if size(B, 2) != n
                throw(DimensionMismatch(lazy"B has second dimension $(size(B,2)) but needs $n"))
            end
            p = size(B, 1)
            k = Ref{BlasInt}()
            l = Ref{BlasInt}()
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ws.alpha = similar(A, $relty, n)
            ws.beta = similar(A, $relty, n)
            ldu = max(1, m)
            ws.U = jobu == 'U' ? similar(A, $elty, ldu, m) : similar(A, $elty, 0, 0)
            ldv = max(1, p)
            ws.V = jobv == 'V' ? similar(A, $elty, ldv, p) : similar(A, $elty, 0, 0)
            ldq = max(1, n)
            ws.Q = jobq == 'Q' ? similar(A, $elty, ldq, n) : similar(A, $elty, 0, 0)
            ws.work = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            cmplx = eltype(A) <: Complex
            ws.iwork = Vector{BlasInt}(undef, n)
            info = Ref{BlasInt}()
            if cmplx
                ws.rwork = Vector{$relty}(undef, 2n)
                ccall((@blasfunc($ggsvd3), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                        Ref{BlasInt}, Clong, Clong, Clong),
                    jobu, jobv, jobq, m,
                    n, p, k, l,
                    A, lda, B, ldb,
                    ws.alpha, ws.beta, ws.U, ldu,
                    ws.V, ldv, ws.Q, ldq,
                    ws.work, lwork, ws.rwork, ws.iwork,
                    info, 1, 1, 1)
            else
                ccall((@blasfunc($ggsvd3), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ptr{$elty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt},
                        Clong, Clong, Clong),
                    jobu, jobv, jobq, m,
                    n, p, k, l,
                    A, lda, B, ldb,
                    ws.alpha, ws.beta, ws.U, ldu,
                    ws.V, ldv, ws.Q, ldq,
                    ws.work, lwork, ws.iwork, info,
                    1, 1, 1)
            end
            lwork = BlasInt(ws.work[1])
            resize!(ws.work, lwork)
            return ws
        end
        GeneralizedSVDWs(A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty}; kwargs...) = Base.resize!(
            GeneralizedSVDWs(Vector{$relty}(undef, 0), Vector{$relty}(undef, 0),
                similar(A, 0, 0), similar(A, 0, 0), similar(A, 0, 0),
                Vector{$elty}(undef, 0), Vector{$relty}(undef, 0), Vector{BlasInt}(
                    undef, 0)),
            A,
            B;
            kwargs...)
        #       SUBROUTINE ZGGSVD( JOBU, JOBV, JOBQ, M, N, P, K, L, A, LDA, B,
        #      $                   LDB, ALPHA, BETA, U, LDU, V, LDV, Q, LDQ, WORK,
        #      $                   RWORK, IWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBQ, JOBU, JOBV
        #       INTEGER            INFO, K, L, LDA, LDB, LDQ, LDU, LDV, M, N, P
        # *     ..
        # *     .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   ALPHA( * ), BETA( * ), RWORK( * )
        #       COMPLEX*16         A( LDA, * ), B( LDB, * ), Q( LDQ, * ),
        #      $                   U( LDU, * ), V( LDV, * ), WORK( * )
        function ggsvd3!(ws::GeneralizedSVDWs, jobu::AbstractChar, jobv::AbstractChar,
                         jobq::AbstractChar, A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty},
                         resize = true)
            require_one_based_indexing(A, B)
            chkstride1(A, B)
            if VERSION >= v"1.11"
                @chkvalidparam 2 jobu ('U', 'N')
                @chkvalidparam 3 jobv ('V', 'N')
                @chkvalidparam 4 jobq ('Q', 'N')
            end
            m, n = size(A)
            if size(B, 2) != n
                throw(DimensionMismatch(lazy"B has second dimension $(size(B,2)) but needs $n"))
            end
            p = size(B, 1)
            k = Ref{BlasInt}()
            l = Ref{BlasInt}()
            lda = max(1, stride(A, 2))
            ldb = max(1, stride(B, 2))
            ldu = max(1, m)
            ldv = max(1, p)
            ldq = max(1, n)
            if jobu == 'U' && size(ws.U) != (ldu, m)
                if resize
                    resize!(ws, A, jobu = jobu, jobv = jobv, jobq = jobq)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            end                
            if jobv == 'V' && size(ws.V) != (ldv, p)
                if resize
                    resize!(ws, A, jobu = jobu, jobv = jobv, jobq = jobq)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            end                
            if jobq == 'Q' && size(ws.Q) != (ldq, n)
                if resize
                    resize!(ws, A, jobu = jobu, jobv = jobv, jobq = jobq)
                else
                    throw(ArgumentError("Workspace has wrong size, use resize!()."))
                end
            end                
            cmplx = eltype(A) <: Complex
            info = Ref{BlasInt}()
            if cmplx
                ccall((@blasfunc($ggsvd3), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ptr{BlasInt},
                        Ptr{BlasInt}, Clong, Clong, Clong),
                    jobu, jobv, jobq, m,
                    n, p, k, l,
                    A, lda, B, ldb,
                    ws.alpha, ws.beta, ws.U, ldu,
                    ws.V, ldv, ws.Q, ldq,
                    ws.work, length(ws.work), ws.rwork, ws.iwork,
                    info, 1, 1, 1)
            else
                ccall((@blasfunc($ggsvd3), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ref{BlasInt},
                        Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$relty}, Ptr{$relty}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt},
                        Ptr{BlasInt}, Clong, Clong, Clong),
                    jobu, jobv, jobq, m,
                    n, p, k, l,
                    A, lda, B, ldb,
                    ws.alpha, ws.beta, ws.U, ldu,
                    ws.V, ldv, ws.Q, ldq,
                    ws.work, length(ws.work), ws.iwork,
                    info, 1, 1, 1)
            end
            chklapackerror(info[])
            if m - k[] - l[] >= 0
                R = triu(A[1:(k[] + l[]), (n - k[] - l[] + 1):n])
            else
                R = triu([A[1:m, (n - k[] - l[] + 1):n];
                          B[(m - k[] + 1):l[], (n - k[] - l[] + 1):n]])
            end
            return ws.U, ws.V, ws.Q, ws.alpha, ws.beta, k[], l[], R
        end
    end
end

"""
    GeneralizedSVDWs(A::AbstractMatrix; jobu = 'U', jobv = 'V', jobq = 'Q')

creates a workspace for [`LinearAlgebra.SVD`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SVD)
factorization using the [`LAPACK.ggsvd3!`](@ref) function.

# Arguments
- `A::AbstractMatrix`: the matrix to be decomposed. Only the shape of the matrix matters for building the workspace. The actual ellements value does not.
- `jobu::AbstractChar`: one of `U` or `N`. Default = 'U'. See [ggsvd3!](@ref).
- `jobv::AbstractChar`: one of `V` or `N`. Default = 'V'. See [ggsvd3!](@ref).
- `jobq::AbstractChar`: one of `Q` or `N`. Default = 'Q'. See [ggsvd3!](@ref).

# Arguments

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> B = [2.2 3.3
            5.2 4.3]
2×2 Matrix{Float64}:
 2.2  3.3
 5.2  4.3

julia> ws = GeneralizedSVDWs(A, B)
GeneralizedSVDWs{Float64, Matrix{Float64}, Float64}
  alpha: 2-element Vector{Float64}
  beta: 2-element Vector{Float64}
  U: 2×2 Matrix{Float64}
  V: 2×2 Matrix{Float64}
  Q: 2×2 Matrix{Float64}
  work: 102-element Vector{Float64}
  rwork: 0-element Vector{Float64}
  iwork: 2-element Vector{Int64}


julia> t = FastLapackInterface.ggsvd3!(ws, 'U', 'V', 'Q', A, B);

julia> GeneralizedSVD(t...)
GeneralizedSVD{Float64, Matrix{Float64}, Float64, Vector{Float64}}
U factor:
2×2 Matrix{Float64}:
 -0.309806  0.9508
  0.9508    0.309806
V factor:
2×2 Matrix{Float64}:
 -0.653816  0.756653
  0.756653  0.653816
Q factor:
2×2 Matrix{Float64}:
  0.723532  0.690291
 -0.690291  0.723532
D1 factor:
2×2 Matrix{Float64}:
 0.911256  0.0
 0.0       0.517359
D2 factor:
2×2 Matrix{Float64}:
 0.411841  0.0
 0.0       0.855768
R0 factor:
2×2 Matrix{Float64}:
 2.54834  6.10941
 0.0      8.57328
```
"""
GeneralizedSVDWs(A::AbstractMatrix; jobu::AbstractChar = 'U', jobv::AbstractChar = 'V', jobq::AbstractChar = 'Q')

"""
    ggsvd3!(ws, jobu, jobv, jobq, A, B; resize = true) -> (U, V, Q, alpha, beta, k, l, R)

Finds the generalized singular value decomposition of `A` and `B`, `U'*A*Q = D1*R`
and `V'*B*Q = D2*R`, using a preallocated [`GeneralizedSVDWs`](@ref).
`D1` has `alpha` on its diagonal and `D2` has `beta` on its
diagonal. If `jobu = U`, the orthogonal/unitary matrix `U` is computed. If
`jobv = V` the orthogonal/unitary matrix `V` is computed. If `jobq = Q`,
the orthogonal/unitary matrix `Q` is computed. If `jobu`, `jobv`, or `jobq` is
`N`, that matrix is not computed. The value of `jobu`, `jobv` and `jobq` must be the same in `GeneralizedSVDWs` and `ggsvd3!`. This function requires LAPACK 3.6.0.
If `ws` does not have the appropriate size for `A`, `B`, and the work to be done,
if `resize=true`, it will be automatically resized accordingly. 
"""
ggsvd3!(ws::GeneralizedSVDWs, jobu::AbstractChar, jobv::AbstractChar, jobq::AbstractChar,
    A::AbstractMatrix, B::AbstractMatrix; resize = true)
