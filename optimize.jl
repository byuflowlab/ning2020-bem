using CCBlade
using FLOWMath
using SpecialFunctions: gamma
import LinearAlgebra: dot
using Snopt
import ForwardDiff
import ReverseDiff
import SparseArrays, SparseDiffTools
import FiniteDiff
import NLsolve


function residual2d(as, aps, rotor, section, op)

    phi = atan((1 + as)*op.Vx/((1 - aps)*op.Vy))
    _, outputs = CCBlade.residual(phi, rotor, section, op)
    
    R = [-outputs.a - as; 10*(-outputs.ap - aps)]

    return R, outputs

end


function solve2d(rotor, section, op)

    # ----- solve residual function ------    

    # wrapper to residual function
    R(x) = residual2d(x[1], x[2], rotor, section, op)[1]

    a0 = -1.0/3
    ap0 = -0.1
    x0 = [a0; ap0]  

  
    results = NLsolve.nlsolve(R, x0, method=:trust_region, ftol=1e-14, autodiff=:forward)
    if !NLsolve.converged(results)
        println("not converged")
        println(rotor)
        println(section)
        println(op)
    end
    x = results.zero
    
    _, outputs = residual2d(x[1], x[2], rotor, section, op)

    return outputs
end


function residualadjoint(R, ymin, ymax, x)
    y, _ = FLOWMath.brent(y -> R(x, y), ymin, ymax)
    return y
end
 
function residualadjoint(R, ymin, ymax, d::Vector{D}) where D <: ForwardDiff.Dual

    # solve root
    x = ForwardDiff.value.(d)
    y = residualadjoint(R, ForwardDiff.value(ymin), ForwardDiff.value(ymax), x)

    # compute derivatives
    wrap(xy) = R(xy[1:end-1], xy[end])
    g = Vector{Float64}(undef, length(x) + 1)
    ForwardDiff.gradient!(g, wrap, [x; y])
    drdx = g[1:end-1]
    drdy = g[end]
    dydx = -drdx/drdy

    # assemble dual number
    b_in = zip(collect.(ForwardDiff.partials.(d))...)
    b_arr = map(x->dot(dydx, x), b_in)
    p = ForwardDiff.Partials((b_arr...,))
    return D(y, p)
end


function solveadjoint(rotor, section, op)

    # error handling
    if typeof(section) <: Vector
        error("You passed in an vector for section, but this funciton does not accept an vector.\nProbably you intended to use broadcasting (notice the dot): solve.(Ref(rotor), sections, ops)")
    end

    # check if we are at hub/tip
    if isapprox(section.r, rotor.Rhub, atol=1e-6) || isapprox(section.r, rotor.Rtip, atol=1e-6)
        return Outputs()  # no loads at hub/tip
    end

    # parameters
    npts = 10  # number of discretization points to find bracket in residual solve

    # unpack
    Vx = op.Vx
    Vy = op.Vy
    theta = section.theta + op.pitch

    # ---- determine quadrants based on case -----
    Vx_is_zero = isapprox(Vx, 0.0, atol=1e-6)
    Vy_is_zero = isapprox(Vy, 0.0, atol=1e-6)

    # quadrants
    epsilon = 1e-6
    q1 = [epsilon, pi/2]
    q2 = [-pi/2, -epsilon]
    q3 = [pi/2, pi-epsilon]
    q4 = [-pi+epsilon, -pi/2]

    if Vx_is_zero && Vy_is_zero
        return Outputs()

    elseif Vx_is_zero

        startfrom90 = false  # start bracket at 0 deg.

        if Vy > 0 && theta > 0
            order = (q1, q2)
        elseif Vy > 0 && theta < 0
            order = (q2, q1)
        elseif Vy < 0 && theta > 0
            order = (q3, q4)
        else  # Vy < 0 && theta < 0
            order = (q4, q3)
        end

    elseif Vy_is_zero

        startfrom90 = true  # start bracket search from 90 deg

        if Vx > 0 && abs(theta) < pi/2
            order = (q1, q3)
        elseif Vx < 0 && abs(theta) < pi/2
            order = (q2, q4)
        elseif Vx > 0 && abs(theta) > pi/2
            order = (q3, q1)
        else  # Vx < 0 && abs(theta) > pi/2
            order = (q4, q2)
        end

    else  # normal case

        startfrom90 = false

        if Vx > 0 && Vy > 0
            order = (q1, q2, q3, q4)
        elseif Vx < 0 && Vy > 0
            order = (q2, q1, q4, q3)
        elseif Vx > 0 && Vy < 0
            order = (q3, q4, q1, q2)
        else  # Vx[i] < 0 && Vy[i] < 0
            order = (q4, q3, q2, q1)
        end

    end

        

    # ----- solve residual function ------

    # # wrapper to residual function to accomodate format required by fzero
    R(phi) = CCBlade.residual(phi, rotor, section, op)[1]

    function R2(x, y)
        section2 = Section(section.r, x[2], x[3], section.af)
        op2 = OperatingPoint(x[4], x[5], op.rho, x[1], op.mu, op.asound)

        return CCBlade.residual(y, rotor, section2, op2)[1]
    end
    x2 = [op.pitch, section.chord, section.theta, op.Vx, op.Vy]

    success = false
    for j = 1:length(order)  # quadrant orders.  In most cases it should find root in first quadrant searched.
        phimin, phimax = order[j]

        # check to see if it would be faster to reverse the bracket search direction
        backwardsearch = false
        if !startfrom90
            if phimin == -pi/2 || phimax == -pi/2  # q2 or q4
                backwardsearch = true
            end
        else
            if phimax == pi/2  # q1
                backwardsearch = true
            end
        end
        
        # force to dual numbers if necessary
        phimin = phimin*one(section.chord)
        phimax = phimax*one(section.chord)

        # find bracket
        success, phiL, phiU = CCBlade.firstbracket(R, phimin, phimax, npts, backwardsearch)

        # once bracket is found, solve root finding problem and compute loads
        if success
            # phistar, _ = FLOWMath.brent(R, phiL, phiU)
            phistar = residualadjoint(R2, phiL, phiU, x2)
            _, outputs = CCBlade.residual(phistar, rotor, section, op)
            return outputs
        end    
    end    

    # it shouldn't get to this point.  if it does it means no solution was found
    # it will return empty outputs
    # alternatively, one could increase npts and try again
    
    @warn "Invalid data (likely) for this section.  Zero loading assumed."
    return Outputs()
end



function shape(cspline, tspline)

    Rhub = 1.5
    Rtip = 63.0
    r = [2.8667, 5.6000, 8.3333, 11.7500, 15.8500, 19.9500, 24.0500,
        28.1500, 32.2500, 36.3500, 40.4500, 44.5500, 48.6500, 52.7500,
        56.1667, 58.9000, 61.6333]

    rspline = [0.00, 0.25, 0.5, 0.75, 1.0]*(Rtip-Rhub) .+ Rhub 
    chord = akima(rspline, cspline, r) 
    
    rspline2 = [0.11111, 0.4, 0.7, 1.0]*(Rtip-Rhub) .+ Rhub 
    theta = [[tspline[1], tspline[1]]; akima(rspline2, tspline, r[3:end])]  # fixed root twist b/c it is a cylinder

    return r, chord, theta, Rhub, Rtip
end


function turbineopt(cspline, tspline, tsr, pitch, method)

    # define pitch to start at zero, add root pitch to twist instead.
    p0 = pitch[1]
    pitch .-= p0
    tspline .+= p0

    # ------- NREL 5MW --------

    B = 3
    turbine = true
    precone = 2.5*pi/180

    r, chord, theta, Rhub, Rtip = shape(cspline, tspline)
    rotor = Rotor(Rhub, Rtip, B; turbine, precone)


    # chord = [3.542, 3.854, 4.167, 4.557, 4.652, 4.458, 4.249, 4.007, 3.748,
    #     3.502, 3.256, 3.010, 2.764, 2.518, 2.313, 2.086, 1.419]
    # theta = pi/180*[13.308, 13.308, 13.308, 13.308, 11.480, 10.162, 9.011, 7.795,
    #     6.544, 5.361, 4.188, 3.125, 2.319, 1.526, 0.863, 0.370, 0.106]

    # Define airfoils.  In this case we have 8 different airfoils that we load into an array.
    # These airfoils are defined in files.
    aftypes = Array{AlphaAF}(undef, 8)
    aftypes[1] = AlphaAF("airfoils/Cylinder1.dat", radians=false)
    aftypes[2] = AlphaAF("airfoils/Cylinder2.dat", radians=false)
    aftypes[3] = AlphaAF("airfoils/DU40_A17.dat", radians=false)
    aftypes[4] = AlphaAF("airfoils/DU35_A17.dat", radians=false)
    aftypes[5] = AlphaAF("airfoils/DU30_A17.dat", radians=false)
    aftypes[6] = AlphaAF("airfoils/DU25_A17.dat", radians=false)
    aftypes[7] = AlphaAF("airfoils/DU21_A17.dat", radians=false)
    aftypes[8] = AlphaAF("airfoils/NACA64_A17.dat", radians=false)

    # indices correspond to which airfoil is used at which station
    af_idx = [1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8, 8, 8, 8, 8, 8]

    # create airfoil array 
    airfoils = aftypes[af_idx]

    sections = Section.(r, chord, theta, airfoils)

    # operating point for the turbine
    yaw = 0.0*pi/180
    tilt = 5.0*pi/180
    hubHt = 90.0
    shearExp = 0.2

    rotorR = Rtip*cos(precone)
    azimuth = 0.0*pi/180
    rho = 1.225  #*one(pitch[1])

    nV = length(pitch)
    Vin = 3.0
    Vout = 25.0
    V = range(Vin, Vout, length=nV)
    # Omega_min = 0.0
    Omega_max = 12.0*pi/30.0
    Omega = min.(collect(V).*tsr/rotorR, Omega_max)  # collect needed for ForwardDiff
    P = similar(V, eltype(chord[1]))
    Q = similar(V, eltype(chord[1]))
    T = similar(V, eltype(chord[1]))

    if method == "traditional"
        solver = solve2d
    elseif method == "adjoint"
        solver = solveadjoint
    else
        solver = solve
    end


    for i = 1:nV
        ops = windturbine_op.(V[i], Omega[i], pitch[i], r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
        outputs = solver.(Ref(rotor), sections, ops)

        T[i], Q[i] = thrusttorque(rotor, sections, outputs)
        P[i] = Q[i]*Omega[i]
    end

    i = 30  # near rated
    ops = windturbine_op.(V[i], Omega[i], zero(rho), r, precone, yaw, tilt, azimuth, hubHt, shearExp, rho)
    outputs = solver.(Ref(rotor), sections, ops)
    Np = outputs.Np

    k = 2.0  # weibull shape
    Vbar = 6.0  # mean speed of weibull
    A = Vbar / gamma(1.0 + 1.0/k)
    cdf = 1.0 .- exp.(-(V/A).^k)
    AEP = trapz(cdf, P) * 365*24

    return AEP, V, T, P, Q, Np
end
# --------------------------





function run(method, nP)

    # starting point
    cspline = [3.542, 4.6035, 3.748, 2.8255, 1.419]
    tspline = [13.308, 8.0, 4.1, 0.05]*pi/180
    tsr = 7.55
    pitch = 20*pi/180*ones(nP)
    
    x0 = [cspline; tspline; tsr; pitch]
    # bounds
    lb = [0.5*ones(5); -5*pi/180*ones(4); 1.0; zeros(nP)]
    ub = [10*ones(5); 20*pi/180*ones(4); 15.0; 30.0*pi/180*ones(nP)]
    lb[1] = 3.542  # lower bound on root chord, for pitch bearing

    Prated = 5e6

    function wrappersparse!(y, x)

        cspline2 = x[1:5]
        tspline2 = x[6:9]
        tsr2 = x[10]
        idx = 10
        pitch2 = x[idx+1:idx+nP]

        AEP, V, T, P, Q, Np = turbineopt(cspline2, tspline2, tsr2, pitch2, method)

        # constraints
        # Qmax = Prated/Omega_max
        Tmax = 600e3
        y .= [P/Prated .- 1; T/Tmax .- 1; pitch2[1:end-1] .- pitch2[2:end]; Np/6500 .- 1]  #; Omega[1:end-1] .- Omega[2:end]]

        return -AEP/1e10  # maximize AEP
    end

    function wrapper!(y, x)
        yv = @view y[2:end]
        y[1] = wrappersparse!(yv, x)
    end


    y = zeros(1 + 2*nP + nP-1 + 17)
    config = ForwardDiff.JacobianConfig(wrapper!, y, x0)
    J = zeros(length(y), length(x0))
    dfdx = zeros(length(x0))

    if method == "sparse"
        ysp = zeros(2*nP + nP-1 + 17)
        Jspstart = zeros(length(ysp), length(x0))
        configsp = ForwardDiff.JacobianConfig(wrappersparse!, ysp, x0)
        ForwardDiff.jacobian!(Jspstart, wrappersparse!, ysp, x0, configsp)
        Jsp = SparseArrays.sparse(Jspstart)  
        colors = SparseDiffTools.matrix_colors(Jsp)
        cache = SparseDiffTools.ForwardColorJacCache(wrappersparse!, x0, dx=ysp, colorvec=colors, sparsity=Jsp)
        
        # setup for AEP derivative
        # Prated = 5e6
        Vin = 3.0
        Vout = 25.0
        Vvec = range(Vin, Vout, length=nP)
        k = 2.0  # weibull shape
        Vbar = 6.0  # mean speed of weibull
        A = Vbar / gamma(1.0 + 1.0/k)
        cdf = 1.0 .- exp.(-(Vvec/A).^k)
    end


    function optimize(x)

        wrapper!(y, x)
        f = y[1]
        g = y[2:end]

        if method == "fd" || method == "traditional"
            FiniteDiff.finite_difference_jacobian!(J, wrapper!, x)
            dfdx .= J[1, :]
            dgdx = J[2:end, :] 
        elseif method == "dense" || method == "adjoint"
            ForwardDiff.jacobian!(J, wrapper!, y, x, config)
            dfdx .= J[1, :]
            dgdx = J[2:end, :] 
        elseif method == "sparse"
            SparseDiffTools.forwarddiff_color_jacobian!(Jsp, wrappersparse!, x, cache)
            dgdx = Jsp
            dPdx = Prated*Jsp[1:nP, :]
            dfdx .= 0.0
            for i = 1:nP-1
                dfdx .+= (cdf[i+1]-cdf[i])*0.5*(dPdx[i, :] + dPdx[i+1, :]) *365*24/-1e10
            end
        end

        fail = false

        return f, g, dfdx, dgdx, fail
    end

    options = Dict{String, Any}()
    options["Derivative option"] = 1
    # options["Verify level"] = 1
    
    xopt, fopt, info = snopt(optimize, x0, lb, ub, options)
    println(info)
    
    return xopt, fopt
end




nP = 80

# cspline = [3.542, 4.6035, 3.748, 2.8255, 1.419]
# tspline = [13.308, 8.0, 4.1, 0.05]*pi/180
# tsr = 7.55
# pitch = 20*pi/180*ones(nP)

# times from snopt output file
xopt, fopt = run("sparse", nP)  # 4.97 seconds  |  5.0 seconds
# xopt, fopt = run("adjoint", nP)  12.35 seconds
# xopt, fopt = run("dense", nP)  #  24.33 seconds 
# xopt, fopt = run("fd", nP)  # 66.65 seconds 
# xopt, fopt = run("traditional", nP)  # 111.86 seconds 

# [3.542, 4.760045358177065, 2.9965715561082655, 2.3398138560921455, 1.2066787649473403, 0.23217027516809968, 0.04756986696388409, -0.007796061979556598, -0.06376494241659889, 9.103517237624589, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.04947636361503202, 0.05797161687843067, 0.06729546915857863, 0.07611472657716972, 0.084545916198257, 0.10238931381185744, 0.12193484245765156, 0.13840114476780663, 0.1528780920046593, 0.1661148262225162, 0.17844270542770568, 0.19005323615677938, 0.2010597330827229, 0.21154712594386327, 0.22161200026472808, 0.23132359234067848, 0.24071807104749224, 0.24983317401699293, 0.2587050277588686, 0.26736142750083625, 0.275818390328649, 0.2840903067458611, 0.29219122920245194, 0.30013125450993156, 0.3079195503900656, 0.31556651476196457, 0.3230785743491701, 0.33045441669536774, 0.33769172890657534, 0.34479738484667255, 0.3517824782971913, 0.35866274762843314, 0.3654573834688694, 0.3721824701081526, 0.3788485527218031, 0.3854667071235377, 0.3920411537844105, 0.3985651750312578, 0.4050348112927709, 0.4114413018179495, 0.4177705818285269, 0.424010858860524, 0.43016182781798307, 0.43623015160814443, 0.44222484500848497, 0.4481533871498382, 0.45402229272571143, 0.4598371291637423, 0.4656041453005771, 0.47133119764929, 0.4770195808895922, 0.4826679312122287, 0.48827517063016895, 0.4938451545400422, 0.4993741840234126, 0.5048596100513677, 0.5103039515039579]
# [3.542, 4.760692986206912, 2.9966783734883844, 2.3398300971260957, 1.2066823140213758, 0.22486275809414039, 0.04025369182394284, -0.015121128599521107, -0.07109139144185356, 9.10344339126887, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.056802086563028875, 0.06529939395248592, 0.0746233170108889, 0.08344266230613664, 0.09187398455874778, 0.10971722887313595, 0.12926238507060298, 0.14572835507190782, 0.16020510529330254, 0.17344180882707613, 0.185769643429235, 0.19738011403457678, 0.20838655097548162, 0.21887391496737257, 0.22893877752369987, 0.23865036183185256, 0.248044834033399, 0.2571599064738573, 0.2660317059572908, 0.2746880750712266, 0.2831450233816709, 0.2914169310236453, 0.2995178525344254, 0.3074578820449117, 0.3152461848941429, 0.32289315744208563, 0.33040522505092784, 0.3377810791699985, 0.34501840647341464, 0.3521240805874463, 0.359109193185985, 0.3659894849247997, 0.37278414016763617, 0.3795092409983166, 0.38617533446656865, 0.392793502288733, 0.3993679619147863, 0.4058919815432788, 0.4123616227379301, 0.4187681259334287, 0.42509741709573906, 0.4313377061687048, 0.4374886904949092, 0.4435570338497668, 0.4495517482205652, 0.45548031214038714, 0.4613492397910199, 0.4671641062716087, 0.4729311514263918, 0.4786582305690257, 0.4843466467170885, 0.4899950254275976, 0.4956022905346512, 0.5011722971998086, 0.5067013431822566, 0.5121867834941963, 0.5176311406324784]
# [3.542, 4.760747241419858, 2.9966625418796187, 2.3398271539260125, 1.2066810840710032, 0.22503666496505845, 0.04042321256815731, -0.014951527586733918, -0.07092131728433311, 9.103418446605376, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.056631997258220754, 0.06512948262810273, 0.0744534159072778, 0.0832727709815761, 0.09170410415865786, 0.10954735239460169, 0.1290925193198949, 0.14555851409497375, 0.16003528352034246, 0.17327199763476067, 0.18559984471682742, 0.1972103276774164, 0.20821677632993488, 0.21870415013865807, 0.22876901949953213, 0.238480610864331, 0.24787509086368356, 0.2569901670989968, 0.26586196647295973, 0.2745183372071666, 0.28297529017334205, 0.2912472029169533, 0.2993481297377437, 0.30728816488127186, 0.31507647343257833, 0.3227234508079986, 0.3302355230935136, 0.33761138092488335, 0.344848711345666, 0.35195438915003024, 0.35893950592318025, 0.36581980273914405, 0.37261446369619056, 0.3793395712226177, 0.38600567202645814, 0.3926238471871718, 0.3991983132541477, 0.4057223359746752, 0.4121919796750226, 0.4185984843296204, 0.42492777583049546, 0.43116806480057174, 0.43731904936252003, 0.44338739364408275, 0.44938210986803434, 0.4553106765514783, 0.4611796077758426, 0.4669944790197102, 0.47276152983256087, 0.47848861472247267, 0.4841770372462505, 0.4898254207202264, 0.4954326906840798, 0.5010027021538521, 0.50653175123733, 0.5120171944853323, 0.5174615544318769]
# [3.542, 4.760640547347785, 2.9966473316771842, 2.3398212666796043, 1.2066945723590519, 0.24268093081920974, 0.05807326350597842, 0.002696531844767569, -0.05326937474340002, 9.103431500321744, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.038982613992732104, 0.04747960751292016, 0.056803519249397874, 0.06562284868821903, 0.07405415281081855, 0.0918974830513985, 0.1114426835897507, 0.1279087208338638, 0.14238550926832594, 0.15562222031103515, 0.16795006081478, 0.1795605349425147, 0.19056698155453236, 0.20105435160205584, 0.21111921382505322, 0.22083080113894826, 0.23022527856463476, 0.23934035714942842, 0.24821216001440732, 0.2568685309560612, 0.265325481370162, 0.273597390326586, 0.28169831135061674, 0.2896383398675135, 0.2974266415611344, 0.3050736121831604, 0.3125856774534014, 0.31996152384512805, 0.32719884212739436, 0.3343045104327401, 0.3412896219185127, 0.3481699148002591, 0.35496457450723745, 0.36168968874323887, 0.36835579897111215, 0.37497398012342953, 0.3815484476991695, 0.3880724695655256, 0.3945421049490535, 0.4009485948898726, 0.4072778717975746, 0.41351814729819303, 0.41966911951255403, 0.4257374513283227, 0.43173215759325456, 0.43766071784905874, 0.4435296460362384, 0.44934451665195557, 0.4551115697167973, 0.4608386561342819, 0.46652707741059735, 0.47217545640139064, 0.47778272243116404, 0.48335273038634896, 0.4888817757682528, 0.49436721548220375, 0.49981156900129603]


cspline = xopt[1:5]
tspline = xopt[6:9]
tsr = xopt[10]
idx = 10
pitch = xopt[idx+1:idx+nP]
# idx += nP
# Omega = xopt[idx+1:idx+nP]

AEP, V, T, P, Q, Np = turbineopt(cspline, tspline, tsr, pitch, "dense")

using PyPlot
close("all")

figure()
plot(V, P/1e6)
xlabel("V (m/s)")
ylabel("P (MW)")

figure()
plot(V, T)
xlabel("V (m/s)")
ylabel("T (N)")

# figure()
# plot(V, Q)

# p0 = pitch[1]
# println("p0 = ", p0*180/pi)
# pitch .-= p0

figure()
plot(V, pitch*180/pi)
xlabel("V (m/s)")
ylabel("pitch (deg)")

# figure()
# plot(V, Omega)


r, chord, theta, Rhub, Rtip = shape(cspline, tspline)

# theta .+= p0

figure()
plot(r, Np)
xlabel("r (m)")
ylabel(L"N^\prime (N/m)")

figure()
plot(r, chord)
xlabel("r (m)")
ylabel("chord (m)")

figure()
plot(r, theta*180/pi)
xlabel("r (m)")
ylabel("theta (deg)")
