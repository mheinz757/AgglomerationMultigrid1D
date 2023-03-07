"""
    legendre_val( x::Real, n::Integer )

Evaluates all the Legendre polynomials up to degree `n` at a point `x`

# Arguments
- `x`: point to evaluate at
- `n`: highest degree of Legendre polynomial to evaluate

# Outputs
- `funVal::AbstractVector`: values of all the Legendre polynomials up to degree `n` at 
    point `x`
"""
function legendre_val( x::Real, n::Integer )
    if n == 0
        funVal = [1.0];
    else
        funVal = [ 1.0, x ];
        for i = 2:n
            push!( funVal, ( (2*i-1) * x * funVal[i] - (i-1) * funVal[i-1] ) / i );
        end
    end

    return funVal
end


"""
    legendre_val_and_deriv( x::Real, n::Integer )

Evaluates values of all the Legendre polynomials up to degree `n` and their derivatives at 
a point `x`

# Arguments
- `x`: point to evaluate at
- `n`: highest degree of Legendre polynomial to evaluate

# Outputs
- `funVal::AbstractVector`: values of all the Legendre polynomials up to degree `n` at 
    point `x`
- `derivVal::AbstractVector`: values of the derivative of all the Legendre polynomials up to 
    degree `n` at point `x`
"""
function legendre_val_and_deriv( x::Real, n::Integer )
    if n == 0
        funVal = [1.0];
        derivVal = [0.0];
    else
        funVal = [ 1.0, x ];
        derivVal = [ 0.0, 1.0 ];
        for i = 2:n
            push!( funVal, ( (2*i-1) * x * funVal[i] - (i-1) * funVal[i-1] ) / i );
            push!( derivVal, (2*i-1) * funVal[i] + derivVal[i-1] );
        end
    end

    return funVal, derivVal
end