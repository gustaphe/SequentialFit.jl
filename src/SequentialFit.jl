module SequentialFit
using LsqFit, RecipesBase, Calculus, Optim, LinearAlgebra

export SeqFit, update!, next_x

import Base.push!

mutable struct SeqFit
    f
    model
    X
    Y
    fit
end
function SeqFit(f, model, X, p0)
    Y = f.(X)
    return SeqFit(f, model, copy(X), Y, curve_fit(model, X, Y, p0))
end
function SeqFit(f, model, p0)
    return SeqFit(f, model, [-1.0, 0.0, 1.0], p0)
end

function push!(s::SeqFit, x...)
    push!(s.X, x...)
    push!(s.Y, s.f.(x)...)
    return s.fit = curve_fit(s.model, s.X, s.Y, s.fit.param)
end

function update!(s::SeqFit)
    x_new = next_x(s)
    return push!(s, x_new)
end

function next_x(s::SeqFit)
    isempty(s.X) && return zero(eltype(s.X))
    o = optimize(
        x -> selectionFunction(x, s),
        ([2 -1; -1 2] * [extrema(s.X)...])..., # cast a net 100 % outside limits
    )
    return o.minimizer
end

function selectionFunction(x, s)
    return -norm(Calculus.gradient(p -> s.model(x, p), s.fit.param))^2 *
           prod(abs.(s.X .- x) .^ 2)
end

@recipe function f(s::SeqFit; groundtruth=false, selection=false)
    @series begin # measurements
        subplot := 1
        seriestype := :scatter
        label --> "Samples"
        s.X, s.Y
    end

    @series begin # fit
        subplot := 1
        seriestype := :line
        label --> "Fit"
        x -> s.model(x, s.fit.param)
    end

    if groundtruth
        @series begin # Ground truth
            subplot := 1
            seriestype := :line
            linestyle --> :dash
            label --> "Ground truth"
            s.f
        end
    end

    if selection
        layout := (2, 1)
        link := :x
        @series begin # Selection function
            subplot := 2
            seriestype := :line
            linestyle --> :dash
            label --> "Selection function"
            ylims := :auto
            yticks := false
            legend := :bottomright
            x -> selectionFunction(x, s)
        end
    end
end

end
