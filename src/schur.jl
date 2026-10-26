import LinearAlgebra.LAPACK: gees!, gges!, tgsen!
## Schur and generalized Schur decomposition with reordering

# SELECT functions
"""
   enum SCHURORDER lp rp id ed

# Keywords
    - `lp`: Left plane (real(eigenvalue) < criterium)
    - `rp`: Right plane (real(eigenvalue) >= criterium)
    - `id`: Interior of disk (abs(eigenvalue)^2 < criterium)
    - `ed`: Exterior of disk (abs(eigenvalue)^2 >= criterium)

# Note
    - the left half-plane is obtained with criterium = 0
    - the unit disk is obtained with criterium = 1
    - because of numerical error in computing repeated eigenvalues, you need to adapt
      criterium depending whether you want to include or not 0 is the left half-plane or
      1 in the unit disk
    - criterium is passed as optional parameter to `gees` and `gges` functions
"""
@enum SCHURORDER lp rp id ed

# store the criterium for preset selection functions

const SCHUR_CRITERIUM = Ref{Float64}()

# preset selection functions
# gees
function lpselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr < SCHUR_CRITERIUM[])
end

function rpselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr > SCHUR_CRITERIUM[])
end

function idselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 < SCHUR_CRITERIUM[])
end

function edselect(wr_::Ptr, wi_::Ptr)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return (wr^2 + wi^2 >= SCHUR_CRITERIUM[])
end

function selectorder2(so::SCHURORDER, criterium)
    global SCHUR_CRITERIUM[] = criterium
    if so == lp
        return @cfunction(lpselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == rp
        return @cfunction(rpselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == id
        return @cfunction(idselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == ed
        return @cfunction(edselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}))
    else
        error("Unknown SCHURDORDER keyword: $so")
    end
end

#gges

function lpselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    return (αr < SCHUR_CRITERIUM[] * β)
end

function rpselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    return (αr >= SCHUR_CRITERIUM[] * β)
end

function idselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    return (αr^2 + αi^2 < SCHUR_CRITERIUM[] * β^2)
end

function edselect(αr_::Ptr, αi_::Ptr, β_::Ptr)::Cint
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    return (αr^2 + αi^2 >= SCHUR_CRITERIUM[] * β^2)
end

function selectorder3(so::SCHURORDER, criterium)
    global SCHUR_CRITERIUM[] = criterium
    if so == lp
        return @cfunction(lpselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == rp
        return @cfunction(rpselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == id
        return @cfunction(idselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    elseif so == ed
        return @cfunction(edselect, Cint,
            (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    else
        error("Unknown SCHURDORDER keyword: $so")
    end
end

# prepares user provided Julia selection functions cfunction
# gees
function schurselect(f::Function, wr_::Ptr, wi_::Ptr)
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return convert(Cint, f(wr, wi) ? 1 : 0)
end

# gges
function schurselect(f::Function, αr_::Ptr, αi_::Ptr, β_::Ptr)
    αr = unsafe_load(αr_)
    αi = unsafe_load(αi_)
    β = unsafe_load(β_)
    return convert(Cint, f(αr, αi, β) ? 1 : 0)
end

"""
    SchurWs

Workspace to be used with the [`LinearAlgebra.Schur`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedSchur) representation
of the Schur decomposition which uses the [`LAPACK.gees!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> ws = SchurWs(A)
SchurWs{Float64}
  work: 68-element Vector{Float64}
  wr: 2-element Vector{Float64}
  wi: 2-element Vector{Float64}
  vs: 2×2 Matrix{Float64}
  sdim: Base.RefValue{Int64}
  bwork: 2-element Vector{Int64}
  eigen_values: 2-element Vector{ComplexF64}

julia> t = Schur(LAPACK.gees!(ws, 'V', A)...)
Schur{Float64, Matrix{Float64}, Vector{Float64}}
T factor:
2×2 Matrix{Float64}:
 -1.6695  -3.9
  0.0      6.1695
Z factor:
2×2 Matrix{Float64}:
 -0.625424  -0.780285
  0.780285  -0.625424
eigenvalues:
2-element Vector{Float64}:
 -1.6695025194532018
  6.169502519453203

julia> Matrix(t)
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3
```
"""
mutable struct SchurWs{T <: AbstractFloat} <: Workspace
    work::Vector{T}
    wr::Vector{T}
    wi::Vector{T}
    vs::Matrix{T}
    sdim::Ref{BlasInt}
    bwork::Vector{BlasInt}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::SchurWs) = length(ws.wr)

for (gees, elty) in ((:dgees_, :Float64),
    (:sgees_, :Float32))
    @eval begin
        function Base.resize!(ws::SchurWs, A::AbstractMatrix{$elty}; work = true)
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            resize!(ws.wr, n)
            resize!(ws.wi, n)
            ws.vs = zeros($elty, n, n)
            resize!(ws.bwork, n)
            resize!(ws.eigen_values, n)
            if work
                info = Ref{BlasInt}()
                ccall((@blasfunc($gees), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{Cvoid}, Ptr{BlasInt}, Clong, Clong),
                    'V', 'N', C_NULL, n,
                    A, max(1, stride(A, 2)), C_NULL, ws.wr,
                    ws.wi, ws.vs, max(size(ws.vs, 1), 1), ws.work,
                    -1, C_NULL, info, 1, 1)

                chklapackerror(info[])

                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end

        SchurWs(A::AbstractMatrix{$elty}) = resize!(
            SchurWs(Vector{$elty}(undef, 1), $elty[], $elty[], Matrix{$elty}(undef, 0, 0),
                Ref{BlasInt}(), BlasInt[], Complex{$elty}[]),
            A)

        function gees!(ws::SchurWs{$elty}, jobvs::AbstractChar,
                A::AbstractMatrix{$elty};
                select::Union{Nothing, Function, SCHURORDER} = nothing,
                criterium::Number = 0.0,
                resize = true)
            require_one_based_indexing(A)
            chkstride1(A)
            n = checksquare(A)
            nws = length(ws)
            if n != nws
                if resize
                    resize!(ws, A; work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            info = Ref{BlasInt}()
            ldvs = max(size(ws.vs, 1), 1)
            lwork = length(ws.work)
            if select isa Function
                if Sys.ARCH == :aarch64
                    throw(ArgumentError("gees: using a select function is not possible on aarch64 architecture"))
                end
                sfunc(wr, wi) = schurselect(select, wr, wi)
                sel_func = @cfunction($(Expr(:$, :sfunc)), Cint,
                    (Ptr{Cdouble}, Ptr{Cdouble}))
                ccall((@blasfunc($gees), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                    jobvs, 'S', sel_func, n,
                    A, max(1, stride(A, 2)), ws.sdim, ws.wr,
                    ws.wi, ws.vs, ldvs, ws.work,
                    lwork, ws.bwork, info)
            elseif select isa SCHURORDER
                sel_func = selectorder2(select, criterium)
                ccall((@blasfunc($gees), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                    jobvs, 'S', sel_func, n,
                    A, max(1, stride(A, 2)), ws.sdim, ws.wr,
                    ws.wi, ws.vs, ldvs, ws.work,
                    lwork, ws.bwork, info)
            else
                ccall((@blasfunc($gees), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid}, Ref{BlasInt},
                        Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{Cvoid}, Ptr{BlasInt}),
                    jobvs, 'N', C_NULL, n,
                    A, max(1, stride(A, 2)), Ref{BlasInt}(), ws.wr,
                    ws.wi, ws.vs, ldvs, ws.work,
                    lwork, C_NULL, info)
            end

            chklapackerror(info[])

            if iszero(ws.wi)
                return A, ws.vs, ws.wr
            else
                @inbounds for i in axes(A, 1)
                    ws.eigen_values[i] = complex(ws.wr[i], ws.wi[i])
                end
                return A, ws.vs, ws.eigen_values
            end
        end
    end
end

"""
    gees!(ws, jobvs, A; select=nothing, criterium=0.0, resize=true) -> (A, vs, ws.eigen_values)

Computes the eigenvalues (`jobvs = N`) or the eigenvalues and Schur
vectors (`jobvs = V`) of matrix `A`, using the preallocated [`SchurWs`](@ref) worspace `ws`. If `ws` is not of the appropriate size and `resize==true` it will be resized for `A`.
`A` is overwritten by its Schur form, and `ws.eigen_values` is overwritten with the eigenvalues.

It is possible to select the eigenvalues appearing in the top left corner of the Schur form:
- by setting the `select` option to one of
    - `lp`: Left plane (real(eigenvalue) < criterium)
    - `rp`: Right plane (real(eigenvalue) >= criterium)
    - `id`: Interior of disk (abs(eigenvalue)^2 < criterium)
    - `ed`: Exterior of disk (abs(eigenvalue)^2 >= criterium)
  and setting `criterium`.
- by setting `select` equal to a function used to sort the eigenvalues during the decomponsition. In this case, the `criterium` keyword isn't used. The function should have the signature `f(wr::T, wi::T) -> Bool`, where `wr` and `wi` are the real and imaginary parts of the eigenvalue, and `T == eltype(A)`. This is not available on aarch64 architecture.

Returns `A`, `vs` containing the Schur vectors, and `ws.eigen_values`.

See also FastLapackInterface.SCHURORDER
"""
gees!(ws::SchurWs, jobvs::AbstractChar, A::AbstractMatrix; kwargs...)

"""
    GeneralizedSchurWs

Workspace to be used with the [`LinearAlgebra.GeneralizedSchur`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.GeneralizedSchur)
representation of the Generalized Schur decomposition which uses the [`LAPACK.gges!`](@ref) function.

# Examples
```jldoctest
julia> A = [1.2 2.3
            6.2 3.3]
2×2 Matrix{Float64}:
 1.2  2.3
 6.2  3.3

julia> B = [8.2 0.3
            1.7 4.3]
2×2 Matrix{Float64}:
 8.2  0.3
 1.7  4.3

julia> ws = GeneralizedSchurWs(A)
GeneralizedSchurWs{Float64}
  work: 90-element Vector{Float64}
  αr: 2-element Vector{Float64}
  αi: 2-element Vector{Float64}
  β: 2-element Vector{Float64}
  vsl: 2×2 Matrix{Float64}
  vsr: 2×2 Matrix{Float64}
  sdim: Base.RefValue{Int64}
  bwork: 2-element Vector{Int64}
  eigen_values: 2-element Vector{ComplexF64}

julia> t = GeneralizedSchur(LAPACK.gges!(ws, 'V','V', A, B)...)
GeneralizedSchur{Float64, Matrix{Float64}, Vector{ComplexF64}, Vector{Float64}}
S factor:
2×2 Matrix{Float64}:
 -1.43796  1.63843
  0.0      7.16295
T factor:
2×2 Matrix{Float64}:
 5.06887  -4.00221
 0.0       6.85558
Q factor:
2×2 Matrix{Float64}:
 -0.857329  0.514769
  0.514769  0.857329
Z factor:
2×2 Matrix{Float64}:
 -0.560266  0.828313
  0.828313  0.560266
α:
2-element Vector{ComplexF64}:
 -1.4379554610733563 + 0.0im
   7.162947865097022 + 0.0im
β:
2-element Vector{Float64}:
 5.068865029631368
 6.855578082442485
```
"""
mutable struct GeneralizedSchurWs{T} <: Workspace
    work::Vector{T}
    αr::Vector{T}
    αi::Vector{T}
    β::Vector{T}
    vsl::Matrix{T}
    vsr::Matrix{T}
    sdim::Ref{BlasInt}
    bwork::Vector{BlasInt}
    eigen_values::Vector{Complex{T}}
end

Base.length(ws::GeneralizedSchurWs) = length(ws.αr)

# look into matlab function
for (gges, elty) in ((:dgges_, :Float64),
    (:sgges_, :Float32))
    @eval begin
        function Base.resize!(ws::GeneralizedSchurWs, A::AbstractMatrix{$elty}; work = true)
            chkstride1(A)
            n = checksquare(A)
            resize!(ws.αr, n)
            resize!(ws.αi, n)
            resize!(ws.β, n)
            resize!(ws.bwork, n)
            resize!(ws.eigen_values, n)
            ws.vsl = zeros($elty, n, n)
            ws.vsr = zeros($elty, n, n)
            if work
                info = Ref{BlasInt}()
                ccall((@blasfunc($gges), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{Cvoid}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                        Ref{BlasInt}, Clong, Clong, Clong),
                    'V', 'V', 'N', C_NULL,
                    n, A, max(1, stride(A, 2)), A,
                    max(1, stride(A, 2)), C_NULL, ws.αr, ws.αi,
                    ws.β, ws.vsl, n, ws.vsr,
                    n, ws.work, -1, C_NULL,
                    info, 1, 1, 1)

                chklapackerror(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
            end
            return ws
        end
        GeneralizedSchurWs(A::AbstractMatrix{$elty}) = resize!(
            GeneralizedSchurWs(Vector{$elty}(undef, 1), $elty[], $elty[], $elty[],
                Matrix{$elty}(undef, 0, 0), Matrix{$elty}(undef, 0, 0),
                Ref{BlasInt}(), BlasInt[], Complex{$elty}[]),
            A)

        function gges!(ws::GeneralizedSchurWs, jobvsl::AbstractChar,
                jobvsr::AbstractChar,
                A::AbstractMatrix{$elty}, B::AbstractMatrix{$elty};
                select::Union{Nothing, Function, SCHURORDER} = nothing,
                criterium::Number = 0,
                resize = true)
            chkstride1(A, B)
            n = checksquare(A)
            m = checksquare(B)
            if n != m
                throw(DimensionMismatch("dimensions of A, ($n,$n), and B, ($m,$m), must match"))
            end
            nws = length(ws)
            if n != nws
                if resize
                    resize!(ws, A; work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end

            info = Ref{BlasInt}()
            ldvsl = size(ws.vsl, 1)
            ldvsr = size(ws.vsr, 1)
            if select isa Function
                if Sys.ARCH == :aarch64
                    throw(ArgumentError("gges: using a select function is not possible on aarch64 architecture"))
                end
                sfunc(αr, αi, β) = schurselect(select, αr, αi, β)
                sel_func = @cfunction($(Expr(:$, :sfunc)), Cint,
                    (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
                ccall((@blasfunc($gges), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                        Ref{BlasInt}, Clong, Clong, Clong),
                    jobvsl, jobvsr, 'S', sel_func,
                    n, A, max(1, stride(A, 2)), B,
                    max(1, stride(B, 2)), ws.sdim, ws.αr, ws.αi,
                    ws.β, ws.vsl, ldvsl, ws.vsr,
                    ldvsr, ws.work, length(ws.work), ws.bwork,
                    info, 1, 1, 1)
            elseif select isa SCHURORDER
                sel_func = selectorder3(select, criterium)
                ccall((@blasfunc($gges), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                        Ref{BlasInt}, Clong, Clong, Clong),
                    jobvsl, jobvsr, 'S', sel_func,
                    n, A, max(1, stride(A, 2)), B,
                    max(1, stride(B, 2)), ws.sdim, ws.αr, ws.αi,
                    ws.β, ws.vsl, ldvsl, ws.vsr,
                    ldvsr, ws.work, length(ws.work), ws.bwork,
                    info, 1, 1, 1)
            else
                ccall((@blasfunc($gges), liblapack), Cvoid,
                    (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty}, Ptr{$elty},
                        Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{Cvoid},
                        Ref{BlasInt}, Clong, Clong, Clong),
                    jobvsl, jobvsr, 'N', C_NULL,
                    n, A, max(1, stride(A, 2)), B,
                    max(1, stride(B, 2)), Ref{BlasInt}(), ws.αr, ws.αi,
                    ws.β, ws.vsl, ldvsl, ws.vsr,
                    ldvsr, ws.work, length(ws.work), C_NULL,
                    info, 1, 1, 1)
            end
            chklapackerror(info[])
            @inbounds for i in axes(A, 1)
                ws.eigen_values[i] = complex(ws.αr[i], ws.αi[i]) / ws.β[i]
            end
            return A, B, complex.(ws.αr, ws.αi), ws.β, ws.vsl, ws.vsr
        end
    end
end

"""
    gges!(ws, jobvsl, jobvsr, A, B; select=nothing, criterium = 0, resize=true) -> (A, B, ws.α, ws.β, ws.vsl, ws.vsr)

Computes the generalized eigenvalues, generalized Schur form, left Schur
vectors (`jobsvl = V`), or right Schur vectors (`jobvsr = V`) of `A` and
`B`, using preallocated [`GeneralizedSchurWs`](@ref) workspace `ws`.
If `ws` is not of the right size, and `resize==true` it will be resized appropriately.

It is possible to select the eigenvalues appearing in the top left corner of the Schur form:
- by setting the `select` option to one of
    - `lp`: Left plane (real(eigenvalue) < criterium)
    - `rp`: Right plane (real(eigenvalue) >= criterium)
    - `id`: Interior of disk (abs(eigenvalue)^2 < criterium)
    - `ed`: Exterior of disk (abs(eigenvalue)^2 >= criterium)
  and setting `criterium`.
- by setting `select` equal to a function used to sort the eigenvalues during the decomposition. In this case, the `criterium` keyword isn't used. The function should have the signature `f(αr::T, αi::T, β::T) -> Bool` where `T == eltype(A)`. An eigenvalue `(αr[j]+αi[j])/β[j]` is selected if `f(αr[j],αi[j],β[j])` is true, i.e. if either one of a complex conjugate pair of eigenvalues is selected, then both complex eigenvalues are selected. This is not available on aarch64 architecture.

The generalized eigenvalues components are returned in `ws.α` and `ws.β` where `ws.α` is a complex vector and `ẁs.β`, a real vector.
The generalized eigenvalues (`ws.α./ws.β`) are returned in `ws.eigen_values`, a complex vector.
The left Schur vectors are returned in `ws.vsl` and the right Schur vectors are returned in `ws.vsr`.

See also FastLapackInterface.SCHURORDER
"""
gges!(
    ws::GeneralizedSchurWs, jobvsl::AbstractChar, jobvsr::AbstractChar, A::AbstractMatrix,
    B::AbstractMatrix; kwargs...)

"""
    GeneralizedOrdSchurWs

Workspace to reorder a generalized Schur decomposition, as
[`LinearAlgebra.ordschur!`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.ordschur!)
does for a `GeneralizedSchur`, using the [`LAPACK.tgsen!`](@ref) function.
Supports `Complex` matrices.

# Examples
```jldoctest
julia> A = [1.0+1.0im 2.0
            0.5im 3.0]
2×2 Matrix{ComplexF64}:
 1.0+1.0im  2.0+0.0im
 0.0+0.5im  3.0+0.0im

julia> B = [2.0 1.0im
            0.0 1.0]
2×2 Matrix{ComplexF64}:
 2.0+0.0im  0.0+1.0im
 0.0+0.0im  1.0+0.0im

julia> ws = GeneralizedOrdSchurWs(A)
GeneralizedOrdSchurWs{ComplexF64}
  work: 1-element Vector{ComplexF64}
  iwork: 1-element Vector{Int64}
  α: 2-element Vector{ComplexF64}
  β: 2-element Vector{ComplexF64}


julia> F = schur(A, B);

julia> LAPACK.tgsen!(ws, LinearAlgebra.BlasInt[0, 1], F.S, F.T, F.Q, F.Z);

julia> ws.α ./ ws.β ≈ reverse(F.α ./ F.β)
true
```
"""
struct GeneralizedOrdSchurWs{T <: Complex} <: Workspace
    work::Vector{T}
    iwork::Vector{BlasInt}
    α::Vector{T}
    β::Vector{T}
end

Base.length(ws::GeneralizedOrdSchurWs) = length(ws.α)

for (tgsen, elty) in ((:ztgsen_, :ComplexF64),
    (:ctgsen_, :ComplexF32))
    @eval begin
        function Base.resize!(ws::GeneralizedOrdSchurWs, A::AbstractMatrix{$elty}; work = true)
            chkstride1(A)
            n = checksquare(A)
            resize!(ws.α, n)
            resize!(ws.β, n)
            if work
                lda = max(1, stride(A, 2))
                select = zeros(BlasInt, n)  # read by the workspace query
                info = Ref{BlasInt}()
                ccall((@blasfunc($tgsen), liblapack), Cvoid,
                    (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                        Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$elty},
                        Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                    0, 1, 1, select,
                    n, A, lda, A,
                    lda, ws.α, ws.β, A,
                    lda, A, lda, Ref{BlasInt}(),
                    C_NULL, C_NULL, C_NULL, ws.work,
                    -1, ws.iwork, -1, info)
                chklapackerror(info[])
                resize!(ws.work, BlasInt(real(ws.work[1])))
                resize!(ws.iwork, ws.iwork[1])
            end
            return ws
        end

        GeneralizedOrdSchurWs(A::AbstractMatrix{$elty}) = resize!(
            GeneralizedOrdSchurWs(Vector{$elty}(undef, 1), Vector{BlasInt}(undef, 1),
                $elty[], $elty[]),
            A)

        function tgsen!(ws::GeneralizedOrdSchurWs, select::AbstractVector{BlasInt},
                S::AbstractMatrix{$elty}, T::AbstractMatrix{$elty},
                Q::AbstractMatrix{$elty}, Z::AbstractMatrix{$elty}; resize = true)
            require_one_based_indexing(select, S, T, Q, Z)
            chkstride1(select, S, T, Q, Z)
            n = checksquare(S)
            if checksquare(T) != n || checksquare(Q) != n || checksquare(Z) != n
                throw(DimensionMismatch("S, T, Q and Z must all be of size ($n,$n)"))
            end
            if length(select) != n
                throw(DimensionMismatch("length of select, $(length(select)), must be $n"))
            end
            nws = length(ws)
            if n != nws
                if resize
                    resize!(ws, S; work = n > nws)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            info = Ref{BlasInt}()
            ccall((@blasfunc($tgsen), liblapack), Cvoid,
                (Ref{BlasInt}, Ref{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                    Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{$elty}, Ptr{$elty}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt}, Ref{BlasInt},
                    Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{$elty},
                    Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}),
                0, 1, 1, select,
                n, S, max(1, stride(S, 2)), T,
                max(1, stride(T, 2)), ws.α, ws.β, Q,
                max(1, stride(Q, 2)), Z, max(1, stride(Z, 2)), Ref{BlasInt}(),
                C_NULL, C_NULL, C_NULL, ws.work,
                length(ws.work), ws.iwork, length(ws.iwork), info)
            chklapackerror(info[])
            return S, T, ws.α, ws.β, Q, Z
        end
    end
end

"""
    tgsen!(ws, select, S, T, Q, Z; resize=true) -> (S, T, ws.α, ws.β, Q, Z)

Reorders the generalized Schur decomposition `(S, T, Q, Z)` of a matrix pair, so that the
eigenvalues selected by `select` (a vector of `0`s and `1`s) move to the top left corner,
using the preallocated [`GeneralizedOrdSchurWs`](@ref) workspace `ws`.
If `ws` is not of the right size, and `resize==true` it will be resized appropriately.
`S`, `T`, `Q` and `Z` are overwritten, and the reordered generalized eigenvalues are `ws.α ./ ws.β`.
This is the same as `LinearAlgebra.LAPACK.tgsen!(select, S, T, Q, Z)`, but without allocations.
"""
tgsen!(ws::GeneralizedOrdSchurWs, select::AbstractVector{BlasInt}, S::AbstractMatrix,
    T::AbstractMatrix, Q::AbstractMatrix, Z::AbstractMatrix; kwargs...)

"""
    GeneralizedEigenvectorsWs

Workspace to compute eigenvectors of a matrix pair in generalized Schur form `(S, P)`,
using the [`tgevc!`](@ref) function. Supports `Complex` matrices.

# Examples
```jldoctest
julia> A = [1.0+1.0im 2.0
            0.5im 3.0]
2×2 Matrix{ComplexF64}:
 1.0+1.0im  2.0+0.0im
 0.0+0.5im  3.0+0.0im

julia> B = [2.0 1.0im
            0.0 1.0]
2×2 Matrix{ComplexF64}:
 2.0+0.0im  0.0+1.0im
 0.0+0.0im  1.0+0.0im

julia> ws = GeneralizedEigenvectorsWs(A)
GeneralizedEigenvectorsWs{ComplexF64, Float64}
  work: 4-element Vector{ComplexF64}
  rwork: 4-element Vector{Float64}

julia> F = schur(A, B);

julia> side, howmny = 'R', 'B';  # Right eigenvectors, with Back-transform with F.Z

julia> VR = copy(F.Z);  # back-transformed right eigenvectors will be stored here

julia> VL = similar(A, 0, 0);  # placeholder, unused with side = 'R'

julia> FastLapackInterface.tgevc!(ws, side, howmny, LinearAlgebra.BlasInt[], F.S, F.T, VL, VR);

julia> A * VR ≈ B * VR * Diagonal(F.α ./ F.β)
true
```
"""
struct GeneralizedEigenvectorsWs{T <: Complex, RT <: AbstractFloat} <: Workspace
    work::Vector{T}
    rwork::Vector{RT}
end

Base.length(ws::GeneralizedEigenvectorsWs) = length(ws.work) ÷ 2

for (tgevc, elty, relty) in ((:ztgevc_, :ComplexF64, :Float64),
    (:ctgevc_, :ComplexF32, :Float32))
    @eval begin
        function Base.resize!(ws::GeneralizedEigenvectorsWs, A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
            resize!(ws.work, 2n)
            resize!(ws.rwork, 2n)
            return ws
        end

        GeneralizedEigenvectorsWs(A::AbstractMatrix{$elty}) =
            resize!(GeneralizedEigenvectorsWs($elty[], $relty[]), A)

        function tgevc!(ws::GeneralizedEigenvectorsWs, side::AbstractChar,
                howmny::AbstractChar, select::AbstractVector{BlasInt},
                S::AbstractMatrix{$elty}, P::AbstractMatrix{$elty},
                VL::AbstractMatrix{$elty}, VR::AbstractMatrix{$elty}; resize = true)
            require_one_based_indexing(select, S, P, VL, VR)
            chkstride1(select, S, P, VL, VR)
            if side ∉ ('L', 'R', 'B')
                throw(ArgumentError("side must be 'L', 'R' or 'B', but $side was passed"))
            end
            if howmny ∉ ('A', 'B', 'S')
                throw(ArgumentError("howmny must be 'A', 'B' or 'S', but $howmny was passed"))
            end
            n = checksquare(S)
            if checksquare(P) != n
                throw(DimensionMismatch("dimensions of S, ($n,$n), and P, $(size(P)), must match"))
            end
            if howmny == 'S' && length(select) != n
                throw(DimensionMismatch("length of select, $(length(select)), must be $n"))
            end
            nws = length(ws)
            if n != nws
                if resize
                    resize!(ws, S)
                else
                    throw(WorkspaceSizeError(nws, n))
                end
            end
            mm = side == 'L' ? size(VL, 2) : side == 'R' ? size(VR, 2) :
                 min(size(VL, 2), size(VR, 2))
            m = Ref{BlasInt}()
            info = Ref{BlasInt}()
            ccall((@blasfunc($tgevc), liblapack), Cvoid,
                (Ref{UInt8}, Ref{UInt8}, Ptr{BlasInt}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ptr{$elty}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                    Ref{BlasInt}, Ptr{BlasInt}, Ptr{$elty}, Ptr{$relty},
                    Ptr{BlasInt}, Clong, Clong),
                side, howmny, select, n,
                S, max(1, stride(S, 2)), P, max(1, stride(P, 2)),
                VL, max(1, stride(VL, 2)), VR, max(1, stride(VR, 2)),
                mm, m, ws.work, ws.rwork,
                info, 1, 1)
            chklapackerror(info[])
            return VL, VR, m[]
        end
    end
end

"""
    tgevc!(ws, side, howmny, select, S, P, VL, VR; resize=true) -> (VL, VR, m)

Computes some or all of the right (`side = 'R'`), left (`side = 'L'`) or both (`side = 'B'`)
eigenvectors of a matrix pair `(S, P)` in generalized Schur form (upper triangular),
using the preallocated [`GeneralizedEigenvectorsWs`](@ref) workspace `ws`.
If `ws` is not of the right size, and `resize==true` it will be resized appropriately.

- `howmny = 'A'`: compute all eigenvectors.
- `howmny = 'B'`: compute all eigenvectors, back-transformed by the matrices passed in `VL`
  and `VR` (e.g. `Q` and `Z` from a generalized Schur decomposition).
- `howmny = 'S'`: compute only the eigenvectors selected by `select` (a vector of `0`s and `1`s).
  `select` is ignored otherwise.

`VL` and `VR` are overwritten with the eigenvectors in their first `m` columns, and returned without
resizing, so they may be views. Wraps LAPACK's `ztgevc`/`ctgevc` (`LinearAlgebra.LAPACK` doesn't.)
"""
tgevc!(ws::GeneralizedEigenvectorsWs, side::AbstractChar, howmny::AbstractChar,
    select::AbstractVector{BlasInt}, S::AbstractMatrix, P::AbstractMatrix,
    VL::AbstractMatrix, VR::AbstractMatrix; kwargs...)
