module LinSolveAlgo

import Base.strides

const libblastrampoline = "libblastrampoline"

using LinearAlgebra
import LinearAlgebra.BlasInt
import LinearAlgebra.BLAS.@blasfunc
import LinearAlgebra.LAPACK: chklapackerror

export LinSolveWs, linsolve_core!, linsolve_core_no_lu!, lu!

struct LinSolveWs{T<:Number,U<:Integer}
    lu::Vector{T}
    ipiv::Vector{BlasInt}
    function LinSolveWs{T,U}(n::U) where {T<:Number,U<:Integer}
        lu = zeros(T, n * n)
        ipiv = zeros(BlasInt, n)
        new(lu, ipiv)
    end
end

# Float64 is the default
LinSolveWs(n) = LinSolveWs{Float64,Int64}(n)

strides(a::Adjoint) = strides(a.parent)

for (getrf, getrs, elty) in (
    (:dgetrf_, :dgetrs_, :Float64),
    (:sgetrf_, :sgetrs_, :Float32),
    (:zgetrf_, :zgetrs_, :ComplexF64),
    (:cgetrf_, :cgetrs_, :ComplexF32),
)
    @eval begin
        function lu!(a::StridedMatrix{$elty}, ws::LinSolveWs)
            copyto!(ws.lu, a)
            mm, nn = size(a)
            m = Ref{BlasInt}(mm)
            n = Ref{BlasInt}(nn)
            # ws.lu isn't a view and has continuous storage
            lda = Ref{BlasInt}(max(1, mm))
            info = Ref{BlasInt}(0)
            ccall(
                (@blasfunc($getrf), libblastrampoline),
                Cvoid,
                (
                    Ref{BlasInt},
                    Ref{BlasInt},
                    Ptr{$elty},
                    Ref{BlasInt},
                    Ptr{BlasInt},
                    Ref{BlasInt},
                ),
                m,
                n,
                ws.lu,
                lda,
                ws.ipiv,
                info,
            )
            if info[] != 0
                chklapackerror(info[])
            end
        end


        t1 = StridedMatrix{$elty}
        t2 = Transpose{$elty,<:StridedMatrix}
        t3 = Adjoint{$elty,<:StridedMatrix}
    end


    for (elty2, transchar) in ((t1, 'N'), (t2, 'T'), (t3, 'C'))

        @eval begin
            function linsolve_core_no_lu!(
                a::$elty2,
                b::StridedVecOrMat{$elty},
                ws::LinSolveWs,
            )
                mm, nn = size(a)
                m = Ref{BlasInt}(mm)
                n = Ref{BlasInt}(nn)
                nhrs = Ref{BlasInt}(size(b, 2))
                # ws.lu isn't a view and has continuous storage
                lda = Ref{BlasInt}(max(1, mm))
                ldb = Ref{BlasInt}(max(1, stride(b, 2)))
                info = Ref{BlasInt}(0)
                ccall(
                    (@blasfunc($getrs), libblastrampoline),
                    Cvoid,
                    (
                        Ref{UInt8},
                        Ref{BlasInt},
                        Ref{BlasInt},
                        Ptr{$elty},
                        Ref{BlasInt},
                        Ptr{BlasInt},
                        Ptr{$elty},
                        Ref{BlasInt},
                        Ref{BlasInt},
                    ),
                    $transchar,
                    n,
                    nhrs,
                    ws.lu,
                    lda,
                    ws.ipiv,
                    b,
                    ldb,
                    info,
                )
                if info[] != 0
                    chklapackerror(info[])
                end
            end

            function linsolve_core!(a::$elty2, b::StridedVecOrMat{$elty}, ws::LinSolveWs)

                lu!(a, ws)
                linsolve_core_no_lu!(a, b, ws)
            end
        end
    end
end

# if a is an adjoint matrix, we compute lu decomposition
# for its parent as the transposition is done at the
# solution stage
lu!(a::Adjoint, ws::LinSolveWs) = lu!(a.parent, ws)

end
