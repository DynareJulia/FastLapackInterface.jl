A = diagm([1, 1, -0.5])


ws = DgeesWs(3)

dgees!(ws, A)
println(ws.eigen_values)

dgees!(ws, A, SchurAlgo.Lhp())
println(ws.eigen_values)
