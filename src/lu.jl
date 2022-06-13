using LinearAlgebra.LAPACK: chkargsok

struct LinSolveWs
    ipiv::Vector{BlasInt}
end
LinSolveWs(n::Int) = LinSolveWs(zeros(BlasInt, n))
LinSolveWs(a::AbstractMatrix) = LinSolveWs(min(size(a)))

for (getrf, elty) in ((:dgetrf_, :Float64),
                      (:sgetrf_, :Float32),
                      (:zgetrf_, :ComplexF64),
                      (:cgetrf_, :ComplexF32))
    @eval begin
        function getrf!(A::AbstractMatrix{$elty}, ws::LinSolveWs)
            @assert min(size(A)) <= length(ws.ipiv) "Allocated Workspace is too small."
            require_one_based_indexing(A)
            chkstride1(A)
            m, n = size(A)
            lda  = max(1, stride(A, 2))
            info = Ref{BlasInt}()
            ccall((@blasfunc($getrf), liblapack), Cvoid,
                  (Ref{BlasInt}, Ref{BlasInt}, Ptr{$elty},
                   Ref{BlasInt}, Ptr{BlasInt}, Ptr{BlasInt}),
                  m, n, A, lda, ws.ipiv, info)
            chkargsok(info[])
            return A, ws.ipiv, info[] #Error code is stored in LU factorization type
        end
    end
end
# No need to reimplement the solve because can just reuse LU from base Julia
