using SequentialFit, LsqFit, Plots
default(; label=nothing)

function benchmark(; noise_level=0.01, k_0=5, K=200, lims=(-5, 5))
    p_true = randn(3 * 1)#randn(3 * rand(1:10))
    p_0 = p_true .+ 0.1 * randn(size(p_true))
    x_0 = collect(range(lims...; length=k_0))

    f(x) = model(x, p_true) .+ noise_level * randn(size(x))

    s = SeqFit(f, model, x_0, p_0, lims)

    p = Array{Float64}(undef, length(p_true), 2)
    error = Array{Float64}(undef, K - k_0, 2)
    for k in (k_0 + 1):K
        update!(s)
        error[k - k_0, 1] = errorcalc(s.fit.param, p_true) #sqrt(sum(abs2, s.fit.param .- p_true))
        x = collect(range(lims...; length=k))
        fit = curve_fit(model, x, f.(x), p_0)
        error[k - k_0, 2] = errorcalc(fit.param, p_true) #sqrt(sum(abs2, fit.param .- p_true))
        p .= [s.fit.param fit.param]
    end

    x = range(lims...; length=100)
    pl1 = plot((k_0 + 1):K, error; color=[1 2], label=["Sequential fit" "Linear space"], yscale=:log10, xguide="k", yguide="error")

    pl2 = plot(x, model(x, p_true); color=:black, label="Ground truth", xguide="x", yguide="f")
    plot!(pl2, x, f(x); color=:gray, label="Noisy signal")
    plot!(pl2, s.X, s.Y; color=1, seriestype=:scatter)
    plot!(pl2, x, model(x, p[:, 1]); color=1, label="Sequential fit")
    plot!(pl2, x, model(x, p[:, 2]); color=2, label="Linear sample fit")
    plot!(pl2; ylims=extrema(model(x, p_true)), yexpand=true)
    return plot(pl1, pl2; layout=(2, 1))
end

function sumofgaussians(x, p)
    return sum(
               exp.(-(x .- m) .^ 2 ./ (0.2w .^ 2)) * a for
               (a, m, w) in zip(p[1:3:end], p[2:3:end], p[3:3:end])
              )
end

function model(x, p)
    return sumofgaussians.(x, Ref(p))
end

function errorcalc(p, p_0)
    (a_0, m_0, w_0) = (p_0[1:3:end], p_0[2:3:end], p_0[3:3:end])
    (a, m, w) = (p[1:3:end], p[2:3:end], p[3:3:end])
    i_0 = sortperm(m_0)
    i = sortperm(m)
    return sqrt(
                sum(abs2, vcat(a[i] .- a_0[i_0], m[i] .- m_0[i_0], w[i] .- w_0[i_0]))
               )
end
