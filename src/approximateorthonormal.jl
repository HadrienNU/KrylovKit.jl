# Definition of an orthonormal basis
"""
    ApproximateOrthonormalBasis{T} <: Basis{T}

A list of vector like objects of type `T` that are approximatelly mutually orthogonal and normalized to
one, representing an orthonormal basis for some subspace (typically a Krylov subspace). See
also [`Basis`](@ref)

Orthonormality of the vectors contained in an instance `b` of `ApproximateOrthonormalBasis`
(i.e. `all(dot(b[i],b[j]) == I[i,j] for i=1:length(b), j=1:length(b))`) is not checked when
elements are added; it is up to the algorithm that constructs `b` to guarantee
orthonormality.

One can easily orthogonalize or orthonormalize a given vector `v` with respect to a
`b::ApproximateOrthonormalBasis` using the functions
[`w, = orthogonalize(v,b,...)`](@ref orthogonalize) or
[`w, = orthonormalize(v,b,...)`](@ref orthonormalize). The resulting vector `w` of the
latter can then be added to `b` using `push!(b, w)`. Note that in place versions
[`orthogonalize!(v, b, ...)`](@ref orthogonalize) or
[`orthonormalize!(v, b, ...)`](@ref orthonormalize) are also available.

Finally, a linear combination of the vectors in `b::ApproximateOrthonormalBasis` can be obtained by
multiplying `b` with a `Vector{<:Number}` using `*` or `mul!` (if the output vector is
already allocated).
"""
mutable struct ApproximateOrthonormalBasis{T,S<:Number} <: Basis{T} # TODO: mutable or not ?
    basis::OrthonormalBasis{T}
    gram::Matrix{S} # The Gram matrix of the basis
end

ApproximateOrthonormalBasis{T,S}() where {T,S} = ApproximateOrthonormalBasis{T}(OrthonormalBasis{T}(), Matrix{S}(undef,0,0))


# Create space for a new matrix, since Matrix are not resizeable
function allocateMatrix(b::ApproximateOrthonormalBasis, k::Int)
    n = length(b.basis)
    new_gram= zeros(eltype(b.gram),k,k)
    new_gram[1:n,1:n] = b.gram
    b.gram = new_gram
    return b.gram
end

# Iterator methods for ApproximateOrthonormalBasis
Base.IteratorSize(::Type{<:ApproximateOrthonormalBasis}) = Base.HasLength()
Base.IteratorEltype(::Type{<:ApproximateOrthonormalBasis}) = Base.HasEltype()

Base.length(b::ApproximateOrthonormalBasis) = length(b.basis)
Base.eltype(b::ApproximateOrthonormalBasis{T,S}) where {T,S} = T

Base.iterate(b::ApproximateOrthonormalBasis) = Base.iterate(b.basis)
Base.iterate(b::ApproximateOrthonormalBasis, state) = Base.iterate(b.basis, state)

Base.getindex(b::ApproximateOrthonormalBasis, i) = getindex(b.basis, i)
Base.setindex!(b::ApproximateOrthonormalBasis, i, q) = setindex!(b.basis, i, q)
Base.firstindex(b::ApproximateOrthonormalBasis) = firstindex(b.basis)
Base.lastindex(b::ApproximateOrthonormalBasis) = lastindex(b.basis)

Base.first(b::ApproximateOrthonormalBasis) = first(b.basis)
Base.last(b::ApproximateOrthonormalBasis) = last(b.basis)


function Base.popfirst!(b::ApproximateOrthonormalBasis)
    popfirst!(b.basis)
    b.gram = b.gram[2:end,2:end]
end

Base.pop!(b::ApproximateOrthonormalBasis) = pop!(b.basis)

function Base.push!(b::ApproximateOrthonormalBasis{T}, q::T) where {T}
    n = length(b)+1
    # Extend gram matrix
    if n > size(b.gram)[1]
        allocateMatrix(b,n)
    end
    # Compute scalar product of the new element with other element of the basis
    for (i, bi) in enumerate(b.basis)
        b.gram[i,n] = dot(bi,q)
        b.gram[n,i] = b.gram[i,n]
    end
    push!(b.basis, q)
    # Then add norm of q
    b.gram[n,n]  = norm(q)
    return b
end

Base.empty!(b::ApproximateOrthonormalBasis) = (empty!(b.basis); empty!(b.gram); return b)

Base.sizehint!(b::ApproximateOrthonormalBasis, k::Int) = (sizehint!(b.basis, k); allocateMatrix(b, k); return b)
Base.resize!(b::ApproximateOrthonormalBasis, k::Int) = (resize!(b.basis, k); allocateMatrix(b, k); return b)

# Multiplication methods with ApproximateOrthonormalBasis
function Base.:*(b::ApproximateOrthonormalBasis, x::AbstractVector)
    y = zero(eltype(x)) * first(b)
    return mul!(y, b, x)
end
LinearAlgebra.mul!(y, b::ApproximateOrthonormalBasis, x::AbstractVector) = unproject!(y, b, x, 1, 0)

"""
    project!(y::AbstractVector, b::ApproximateOrthonormalBasis, x,
        [α::Number = 1, β::Number = 0, r = Base.OneTo(length(b))])

For a given basis `b`, compute the expansion coefficients `y` resulting from
projecting the vector `x` onto the subspace spanned by `b`; more specifically this computes

```
    y[j] = β*y[j] + α * dot(b[r[j]], x)
```

for all ``j ∈ r``.
"""
function project!(
    y::AbstractVector,
    b::ApproximateOrthonormalBasis,
    x,
    α::Number = true,
    β::Number = false,
    r = Base.OneTo(length(b))
)
    project!(y,b.basis,x,α,β,r)
    y = view(b.gram,1:length(b),1:length(b)) \ y
    return y
end

"""
    unproject!(y, b::ApproximateOrthonormalBasis, x::AbstractVector,
        [α::Number = 1, β::Number = 0, r = Base.OneTo(length(b))])

For a given orthonormal basis `b`, reconstruct the vector-like object `y` that is defined by
expansion coefficients with respect to the basis vectors in `b` in `x`; more specifically
this computes

```
    y = β*y + α * sum(b[r[i]]*x[i] for i = 1:length(r))
```
"""
function unproject!(
    y,
    b::ApproximateOrthonormalBasis,
    x::AbstractVector,
    α::Number = true,
    β::Number = false,
    r = Base.OneTo(length(b))
)
    x = view(b.gram,1:length(b),1:length(b)) * x
    return unproject!(y,b.basis, x,α,β,r)
end


"""
    basistransform!(b::ApproximateOrthonormalBasis, U::AbstractMatrix)

Transform the basis `b` by the matrix `U`. For `b` an basis,
the matrix `U` should be real orthogonal or complex unitary; it is up to the user to ensure
this condition is satisfied. The new basis vectors are given by

```
    b[j] ← b[i] * U[i,j]
```

and are stored in `b`, so the old basis vectors are thrown away. Note that, by definition,
the subspace spanned by these basis vectors is exactly the same.
"""
function basistransform!(b::ApproximateOrthonormalBasis{T}, U::AbstractMatrix) where {T} # U should be unitary or isometric
    #TODO : Transform Gram Matrix
    #U*gram*UT
    b.gram = U'*b.gram*U
    #Transform basis
    basistransform!(b.basis, U)
    return b
end

# Orthogonalization of a vector against a given ApproximateOrthonormalBasis

function orthogonalize!(v::T, b::ApproximateOrthonormalBasis{T}, alg::Orthogonalizer) where {T}
    S = promote_type(eltype(v), eltype(T))
    c = Vector{S}(undef, length(b))
    return orthogonalize!(v, b, c, alg)
end

function orthogonalize!(
    v::T,
    b::ApproximateOrthonormalBasis{T},
    x::AbstractVector,
    ::CompensatedGramSchmidt
) where {T}
    x = project!(x, b, v)
    v = unproject!(v, b, x, -1, 1)
    return (v, x)
end
function reorthogonalize!(
    v::T,
    b::ApproximateOrthonormalBasis{T},
    x::AbstractVector,
    ::CompensatedGramSchmidt
) where {T}
    s = similar(x) ## EXTRA ALLOCATION
    s = project!(s, b, v)
    v = unproject!(v, b, s, -1, 1)
    x .+= s
    return (v, x)
end
function orthogonalize!(
    v::T,
    b::ApproximateOrthonormalBasis{T},
    x::AbstractVector,
    ::CompensatedGramSchmidt2
) where {T}
    (v, x) = orthogonalize!(v, b, x, CompensatedGramSchmidt())
    return reorthogonalize!(v, b, x, CompensatedGramSchmidt())
end
function orthogonalize!(
    v::T,
    b::ApproximateOrthonormalBasis{T},
    x::AbstractVector,
    alg::CompensatedGramSchmidtIR
) where {T}
    nold = norm(v)
    orthogonalize!(v, b, x, CompensatedGramSchmidt())
    nnew = norm(v)
    while eps(one(nnew)) < nnew < alg.η * nold
        nold = nnew
        (v, x) = reorthogonalize!(v, b, x, CompensatedGramSchmidt())
        nnew = norm(v)
    end
    return (v, x)
end
