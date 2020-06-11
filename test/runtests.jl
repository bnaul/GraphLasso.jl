using GraphLasso, DataFrames, Distributions, RDatasets
using LinearAlgebra
using Test


@testset "Test Graph Lasso" begin
    S_iris = cov(Matrix(dataset("datasets", "iris")[:,1:4]))
    α = 1.0
    W_R = [1.68569351 0.000000 0.2743155 0.01969989
           0.00000000 1.189979 0.0000000 0.00000000
           0.27431545 0.000000 4.1162779 0.29560940
           0.01969989 0.000000 0.2956094 1.58100626]
    Θ_R = [ 0.59973155 0.0000000 -0.03996709  0.00000000
            0.00000000 0.8403507  0.00000000  0.00000000
           -0.03996708 0.0000000  0.24890787 -0.04604166
            0.00000000 0.0000000 -0.04604166  0.64111722]
    W_iris, Θ_iris = graphlasso(S_iris, α; tol=1e-6, penalize_diag=true)
    @test norm(W_R-W_iris) < 1e-5
    @test norm(Θ_R-Θ_iris) < 1e-5
end