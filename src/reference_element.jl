struct ReferenceElement
    mP::Int64
    mNodesX::Vector{Float64} # Matrix in 2d

    mGaussQuadNodes::Vector{Float64} # Matrix in 2d
    mGaussQuadWeights::Vector{Float64}

    mBasisFunCoeff::Matrix{Float64} # columns correspond to one basis function
    mBasisGQFunVal::Matrix{Float64} # columns correspond to one basis function
    mBasisGQDerivVal::Matrix{Float64} # columns correspond to one basis function, vector of Matrices in 2d

    mMassMatrix::Matrix{Float64}
end

function ReferenceElement( mP )

    mNodesX = zeros( Float64, mP + 1 );
    if mP >= 1
        mNodesX[1:2] = [-1.0, 1.0];
        mNodesX[3:(mP+1)] = cos.( pi * ( 1:(mP-1) ) / mP );
    else
        mNodesX[1] = 0.0;
    end

    V = zeros( mP+1, mP+1 );
    for i = 1:(mP+1)
        V[i,:] = legendre_val( mNodesX[i], mP );
    end
    mBasisFunCoeff = inv(V);

    mGaussQuadNodes, mGaussQuadWeights = gauss_quad( 2*mP );
    mBasisGQFunVal, mBasisGQDerivVal = evaluate_nodal_basis_fun_and_deriv( mBasisFunCoeff, 
        mGaussQuadNodes );

    mMassMatrix = zeros( mP + 1, mP + 1 );

    for j = 1:(mP + 1)
        for i = 1:j
            for l = 1:length( mGaussQuadWeights )
                mMassMatrix[i,j] += mGaussQuadWeights[l] * mBasisGQFunVal[l,i] * 
                    mBasisGQFunVal[l,j];
            end
        end
    end

    for j = 1:(mP+1)
        for i = (j+1):(mP+1)
            mMassMatrix[i,j] = mMassMatrix[j,i];
        end
    end

    return ReferenceElement( mP, mNodesX, mGaussQuadNodes, mGaussQuadWeights, 
        mBasisFunCoeff, mBasisGQFunVal, mBasisGQDerivVal, mMassMatrix );
end

############################################################################################
############################################################################################
############################################################################################

function evaluate_nodal_basis_fun( basisFunCoeff, nodes )

    basisFunVal = zeros( length( nodes ), size( basisFunCoeff, 1 ) );
    p = size( basisFunCoeff, 1 ) - 1;

    for (l, x) in enumerate( nodes )
        legendreVal = legendre_val( x, p );
        for i = 1:(p+1)
            basisFunVal[l,i] = la.dot( basisFunCoeff[:, i], legendreVal );
        end
    end

    return basisFunVal;
end

function evaluate_nodal_basis_fun_and_deriv( basisFunCoeff, nodes )

    basisFunVal = zeros( length( nodes ), size( basisFunCoeff, 1 ) );
    basisDerivVal = zeros( length( nodes ), size( basisFunCoeff, 1 ) );
    p = size( basisFunCoeff, 1 ) - 1;

    for (l, x) in enumerate( nodes )
        legendreVal, legendreDeriv = legendre_val_and_deriv( x, p );
        for i = 1:(p+1)
            basisFunVal[l,i] = la.dot( basisFunCoeff[:, i], legendreVal );
            basisDerivVal[l,i] = la.dot( basisFunCoeff[:, i], legendreDeriv );
        end
    end

    return basisFunVal, basisDerivVal;
end
