module SequentialOptimization
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
function SeqFit(f,model,X,p0)
    Y = f.(X)
    SeqFit(f,model,copy(X),Y,curve_fit(model,X,Y,p0))
end
SeqFit(f,model,p0) = SeqFit(f,model,Float64[],p0)

function push!(s::SeqFit,x...)
    push!(s.X,x...)
    push!(s.Y,s.f.(x)...)
    s.fit = curve_fit(s.model,s.X,s.Y,s.fit.param)
end

function update!(s::SeqFit)
    x_new = next_x(s)
    push!(s,x_new)
end

function next_x(s::SeqFit)
    isempty(s.X) && return zero(eltype(s.X))
    o = optimize(x -> selectionFunction(x,s),
        ([2 -1;-1 2]*[extrema(s.X)...])... # cast a net 100 % outside limits
    )
    return o.minimizer
end

function selectionFunction(x,s)
    -norm(Calculus.gradient(p->s.model(x,p),s.fit.param))^2*prod(abs.(s.X .- x).^2)
end

@recipe function f(s::SeqFit; groundtruth=false, selection=false)
    legend --> false
    @series begin # measurements
        seriestype:=:scatter
        s.X, s.Y
    end

    @series begin # fit
        seriestype:=:line
        x->s.model(x,s.fit.param)
    end

    if groundtruth
        @series begin # Ground truth
            seriestype:=:line
            s.f
        end
    end

    if selection
        @series begin # Selection function
            seriestype:=:line
            x->selectionFunction(x,s)
        end
    end
end

end
