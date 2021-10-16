A = diagm([1, -0.5, 1])


ws = DgeesWs(3)

dgees!(ws, A)
println(ws.eigen_values)

dgees!(ws, A, >=, 1.0)
println(ws.eigen_values)
