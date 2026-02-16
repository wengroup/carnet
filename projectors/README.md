# unit_vector_projector.yaml

This file contains the symbolic and numerical values of the projector H to construct 
natural tensors from a unit vector. 

The data is organized in the following way:

{rank-normalization:
    {"H_symbolic": symbolic expression for the projector H,
    "H_numerical": numerical values of the projector H,
    "rule": einsum rule to apply the projector H to get the natural tensor from the unit vector
    }
}

- In "rank-normalization", rank is a positive integer that indicates the rank of the
  natural tensor to be constructed, and normalization can be `none` or `unity`,
  indicating whether the projector H is normalized or not.


# tensor_product_projector.yaml

This file contains the symbolic and numerical values of the projector H that performs 
the tensor product of two natural tensors to get a new natural tensor.

The data is organized in the following way:

{l1-l2-l3-normalization:
    {"H_symbolic": symbolic expression for the projector H,
    "H_numerical": numerical values of the projector H,
    "rule": einsum rule to apply the projector H to get the natural tensor from the two input natural tensors
    }
}

- In "l1-l2-l3-normalization", l1 and l2 are the ranks of the two input natural tensors,
  l3 is the rank of the output natural tensor, and normalization can be `none` or
  `unity`, indicating whether the projector H is normalized or not.


# decomposition_and_reconstruction_projector.yaml

This file contains the symbolic and numerical values of the projector G and H to 
decompose and reconstruct physical tensors into and from natural tensors.

The data is organized in the following way:

{physical_tensor_name:
    {"rank": rank of the physical tensor,
     "symmetry": symmetry of the physical tensor,
     "GHS":
        weight: 
            {"G": [{"symbolic": symbolic expression for the projector G,
                   "numerical": numerical values of the projector G }],
             "H": [{"symbolic": symbolic expression for the projector H,
                   "numerical": numerical values of the projector H }],
             "S": [{"symbolic": symbolic expression for the projector S,]
                   "numerical": numerical values of the projector S }]
            }   
    }    
}

- "physical_tensor_name" is the name of the physical tensor, such as "polarizability" "
elasticity".
- "weight" is the weight of the natural tensor in the decomposition of the physical 
  tensor, which is a positive integer.
