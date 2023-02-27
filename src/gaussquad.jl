"""
    function gauss_quad(p)

Gaussian quadrature on [-1,1] for given degree of precision `p`
"""
function gauss_quad(p)
    n = ceil((p+1)/2);
    b = 1:n-1;
    b = @. b / sqrt(4*b^2 - 1);
    eval, evec = la.eigen(la.diagm(1 => b, -1 => b));

    return eval, 2*evec[1,:].^2;
end