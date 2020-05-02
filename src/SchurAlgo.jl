module SchurAlgo
# general Schur decomposition with reordering
# adataped from ./base/linalg/lapack.jl

include("exceptions.jl")

import LinearAlgebra: USE_BLAS64, LAPACKException
import LinearAlgebra: BlasInt, BlasFloat, checksquare, chkstride1 
import LinearAlgebra.BLAS: @blasfunc, libblas
import LinearAlgebra.LAPACK: liblapack, chklapackerror
import Base: has_offset_axes

export DgeesWs, dgees!, DggesWs, dgges!

const criterium = 1+1e-6

#=
function mycompare(wr_, wi_)::Cint
    wr = unsafe_load(wr_)
    wi = unsafe_load(wi_)
    return convert(Cint, ((wr*wr + wi*wi) < criterium) ? 1 : 0)
end
=#

function mycompare(alphar_::Ptr{T}, alphai_::Ptr{T}, beta_::Ptr{T})::Cint where T
    alphar = unsafe_load(alphar_)
    alphai = unsafe_load(alphai_)
    beta = unsafe_load(beta_)
    return convert(Cint, ((alphar*alphar + alphai*alphai) < criterium*beta*beta) ? 1 : 0)
end

mutable struct DgeesWs
    jobvs::Ref{UInt8}
    sdim::Ref{BlasInt}
    wr::Vector{Float64}
    wi::Vector{Float64}
    ldvs::Ref{BlasInt}
    vs::Matrix{Float64}
    work::Vector{Float64}
    lwork::Ref{BlasInt}
    bwork::Vector{Int64}
    eigen_values::Vector{Complex{Float64}}
    info::Ref{BlasInt}

    function DgeesWs(jobvs::Ref{UInt8}, A::StridedMatrix{Float64}, sdim::Ref{BlasInt},
                      wr::Vector{Float64}, wi::Vector{Float64}, ldvs::Ref{BlasInt}, vs::Matrix{Float64},
                      work::Vector{Float64}, lwork::Ref{BlasInt}, bwork::Vector{Int64},
                      eigen_values::Vector{Complex{Float64}}, info::Ref{BlasInt})
        n = Ref{BlasInt}(size(A,1))
        RldA = Ref{BlasInt}(max(1,stride(A,2)))
        Rsort = Ref{UInt8}('N')
#        mycompare_c = @cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}))
        ccall((@blasfunc(dgees_), liblapack), Nothing,
              (Ref{UInt8}, Ref{UInt8}, Ptr{Nothing},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
               Ptr{BlasInt}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt},
               Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt},
               Ptr{BlasInt}),
              jobvs, 'N', C_NULL,
              n, A, RldA, 
              sdim, wr, wi,
              vs, ldvs,
              work, lwork, bwork,
              info)
        chklapackerror(info[])
        lwork = Ref{BlasInt}(real(work[1]))
        work = Vector{Float64}(undef, lwork[])
        new(jobvs, sdim, wr, wi, ldvs, vs, work, lwork, bwork, eigen_values, info)
    end

end


function DgeesWs(A::StridedMatrix{Float64})
    chkstride1(A)
    n, = checksquare(A)
    jobvs = Ref{UInt8}('V')
    sdim = Ref{BlasInt}(0)
    wr = Vector{Float64}(undef, n)
    wi = Vector{Float64}(undef, n)
    ldvs = Ref{BlasInt}(jobvs[] == UInt32('V') ? n : 1)
    vs = Matrix{Float64}(undef, ldvs[], n)
    work = Vector{Float64}(undef, 1)
    lwork = Ref{BlasInt}(-1)
    bwork = Vector{Int64}(undef, n)
    eigen_values = Vector{Complex{Float64}}(undef, n)
    info = Ref{BlasInt}(0)
    DgeesWs(jobvs, A, sdim, wr, wi, ldvs, vs, work, lwork, bwork, eigen_values, info)

end

function DgeesWs(n::Int64)
    A = zeros(n,n)
    DgeesWs(A)
end

function dgees!(ws::DgeesWs,A::StridedMatrix{Float64})
    n = Ref{BlasInt}(size(A,1))
    RldA = Ref{BlasInt}(max(1,stride(A,2)))
    myfunc::Function = make_select_function(>=, 1.0)
    ccall((@blasfunc(dgees_), liblapack), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt},
           Ptr{BlasInt}),
          ws.jobvs, 'N', C_NULL,
          n, A, RldA,
          ws.sdim, ws.wr, ws.wi,
          ws.vs, ws.ldvs,
          ws.work, ws.lwork, ws.bwork,
          ws.info)
    copyto!(ws.eigen_values, complex.(ws.wr, ws.wi))
    chklapackerror(ws.info[])
end
            
function make_select_function(op, crit)::Function
    mycompare = function(wr_, wi_)
        wr = unsafe_load(wr_)
        wi = unsafe_load(wi_)
        return convert(Cint, op(wr*wr + wi*wi, crit) ? 1 : 0)
    end
    return mycompare
end

function dgees!(ws::DgeesWs, A::StridedMatrix{Float64}, op, crit)
    n = Ref{BlasInt}(size(A,1))
    RldA = Ref{BlasInt}(max(1,stride(A,2)))
    myfunc::Function = make_select_function(op, crit)
    mycompare_c = @cfunction($myfunc, Cint, (Ptr{Cdouble}, Ptr{Cdouble}))
    ccall((@blasfunc(dgees_), liblapack), Cvoid,
          (Ref{UInt8}, Ref{UInt8}, Ptr{Cvoid},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt},
           Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ref{BlasInt},
           Ptr{Float64}, Ref{BlasInt}, Ptr{BlasInt},
           Ptr{BlasInt}),
          ws.jobvs, 'S', mycompare_c,
          n, A, RldA,
          ws.sdim, ws.wr, ws.wi,
          ws.vs, ws.ldvs,
          ws.work, ws.lwork, ws.bwork,
          ws.info)
    copyto!(ws.eigen_values, complex.(ws.wr, ws.wi))
    chklapackerror(ws.info[])
end
            
mutable struct DggesWs
    alphar::Vector{Float64}
    alphai::Vector{Float64}
    beta::Vector{Float64}
    lwork::BlasInt
    work::Vector{Float64}
    bwork::Vector{Int64}
    sdim::BlasInt

    function DggesWs(jobvsl::Ref{UInt8}, jobvsr::Ref{UInt8}, sort::Ref{UInt8}, A::StridedMatrix{Float64}, B::StridedMatrix{Float64})
        chkstride1(A, B)
        n, m = checksquare(A, B)
        if n != m
            throw(DimensionMismatch("Dimensions of A, ($n,$n), and B, ($m,$m), must match"))
        end
        n = BlasInt(size(A,1))
        alphar = Vector{Float64}(undef, n)
        alphai = Vector{Float64}(undef, n)
        beta = Vector{Float64}(undef, n)
        bwork = Vector{Int64}(undef, n)
        ldvsl = BlasInt(1)
        ldvsr = BlasInt(1)
        sdim = BlasInt(0)
        lwork = BlasInt(-1)
        work = Vector{Float64}(undef, 1)
        sdim = BlasInt(0)
        info = BlasInt(0)
        mycompare_g_c = @cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
        ccall((@blasfunc(dgges_), liblapack), Nothing,
              (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Nothing},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ref{BlasInt}, Ref{BlasInt}, Ptr{Float64}, Ptr{Float64},
               Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
               Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
               Ref{BlasInt}),
              jobvsl, jobvsr, sort, mycompare_g_c,
              n, A, max(1,stride(A, 2)), B,
              max(1,stride(B, 2)), sdim, alphar, alphai,
              beta, C_NULL, ldvsl, C_NULL,
              ldvsr, work, lwork, bwork,
              info)
        chklapackerror(info)
        lwork = BlasInt(real(work[1]))
        work = Vector{Float64}(undef, lwork)
        new(alphar,alphai,beta,lwork,work,bwork,sdim)
    end
end

function DggesWs(A::StridedMatrix{Float64}, B::StridedMatrix{Float64})
    DggesWs(Ref{UInt8}('N'), Ref{UInt8}('N'), Ref{UInt8}('N'), A, B)
end

function dgges!(jobvsl::Char, jobvsr::Char, A::StridedMatrix{Float64}, B::StridedMatrix{Float64},
                vsl::Matrix{Float64}, vsr::Matrix{Float64}, eigval::Array{ComplexF64,1},
                ws::DggesWs)
    n = size(A,1)
    ldvsl = jobvsl == 'V' ? n : 1
    ldvsr = jobvsr == 'V' ? n : 1
    sort = 'S'
    sdim = Ref{BlasInt}(0)
    info = Ref{BlasInt}(0)
    mycompare_g_c = @cfunction(mycompare, Cint, (Ptr{Cdouble}, Ptr{Cdouble}, Ptr{Cdouble}))
    ccall((@blasfunc(dgges_), liblapack), Nothing,
          (Ref{UInt8}, Ref{UInt8}, Ref{UInt8}, Ptr{Nothing},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ref{BlasInt}, Ptr{BlasInt}, Ptr{Float64}, Ptr{Float64},
           Ptr{Float64}, Ptr{Float64}, Ref{BlasInt}, Ptr{Float64},
           Ref{BlasInt}, Ptr{Float64}, Ref{BlasInt}, Ptr{Int64},
           Ref{Int64}),
          jobvsl, jobvsr, sort, mycompare_g_c,
          n, A, max(1,stride(A, 2)), B,
          max(1,stride(B, 2)), sdim, ws.alphar, ws.alphai,
          ws.beta, vsl, ldvsl, vsr,
          ldvsr, ws.work, ws.lwork, ws.bwork,
          info)
    ws.sdim = sdim[]
    if info[] > 0
        throw(DggesException(info[]))
    end
    for i in 1:n
        eigval[i] = complex(ws.alphar[i],ws.alphai[i])/ws.beta[i]
    end
end

end
