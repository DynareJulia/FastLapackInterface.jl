n = 10

#for elty in (Float32, Float64, ComplexF32, ComplexF64)

#    A0 = randn(elty, n, n)
A0 = randn(n, n)

A = copy(A0)
ws = QrpWs(A)

geqp3!(A, ws)

@static if VERSION < v"1.7"
    target = qr(A0, Val(true))
else
    target = qr(A0, ColumnNorm())
end

A = randn(n, 15)
vA = zeros(n, 2)
vA .= view(A, :, [7, 9])
ws = QrWs(vA)
geqrf_core!(vA, ws)
ormqr_core!('L', vA', A, ws)
