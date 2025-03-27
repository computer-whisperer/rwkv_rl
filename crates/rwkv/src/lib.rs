#![recursion_limit = "256"]

use burn::prelude::{Tensor};
use burn::tensor::ops::FloatTensor;
use burn::tensor::TensorPrimitive;

pub mod sampling;
pub mod context_manager;
pub mod rwkv7;
pub mod kernel;

#[cfg(test)]
mod test_accuracy;

pub trait RWKVFusedBackend: burn::tensor::backend::Backend {
    fn simple_matmul_test(lhs: FloatTensor<Self>, rhs: FloatTensor<Self>) -> FloatTensor<Self>;
}

pub trait RWKVAutodiffBackend: RWKVFusedBackend + burn::tensor::backend::AutodiffBackend {}

pub fn simple_matmul_test_custom<B: RWKVFusedBackend>(lhs: Tensor<B, 3>, rhs: Tensor<B, 3>) -> Tensor<B, 3> {
    let output = B::simple_matmul_test(lhs.into_primitive().tensor(), rhs.into_primitive().tensor());
    Tensor::from_primitive(TensorPrimitive::Float(output))
}

use burn::tensor::Shape;
use burn_cubecl::{
    CubeBackend, CubeRuntime, FloatElement, IntElement, element::BoolElement,
    kernel::into_contiguous, tensor::CubeTensor,
};
use cubecl::{CubeCount, CubeDim};

/// Implement our custom backend trait for the generic `CubeBackend`.
impl<R: CubeRuntime, F: FloatElement, I: IntElement, BT: BoolElement> RWKVFusedBackend
for CubeBackend<R, F, I, BT>
{
    fn simple_matmul_test(
        lhs: FloatTensor<Self>,
        rhs: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        // Define cube dim, hardcoded for simplicity.
        let cube_dim = CubeDim { x: 16, y: 16, z: 1 };

        lhs.assert_is_on_same_device(&rhs);

        // For simplicity, make sure each tensor is continuous.
        let lhs = into_contiguous(lhs);
        let rhs = into_contiguous(rhs);

        // Get the matmul relevant shapes.
        let ndims = lhs.shape.num_dims();
        let num_rows = lhs.shape.dims[ndims - 2];
        let num_cols = rhs.shape.dims[ndims - 1];

        // Compute shape of output, while tracking number of batches.
        let mut num_batches = 1;
        let mut shape_out = vec![0; ndims];
        for i in shape_out.clone().into_iter().take(ndims - 2) {
            shape_out[i] = usize::max(lhs.shape.dims[i], rhs.shape.dims[i]);
            num_batches *= shape_out[i];
        }
        shape_out[ndims - 2] = num_rows;
        shape_out[ndims - 1] = num_cols;
        let shape_out = Shape::from(shape_out);

        // Create a buffer for the output tensor.
        let buffer = lhs
            .client
            .empty(shape_out.num_elements() * core::mem::size_of::<F>());

        // Create the output tensor primitive.
        let output = CubeTensor::new_contiguous(
            lhs.client.clone(),
            lhs.device.clone(),
            shape_out,
            buffer,
            F::dtype(),
        );

        // Declare the wgsl workgroup with the number of cubes in x, y and z.
        let cubes_needed_in_x = f32::ceil(num_rows as f32 / cube_dim.x as f32) as u32;
        let cubes_needed_in_y = f32::ceil(num_cols as f32 / cube_dim.y as f32) as u32;
        let cube_count =
            CubeCount::Static(cubes_needed_in_x, cubes_needed_in_y, num_batches as u32);

        // Execute lazily the kernel with the launch information and the given buffers. For
        // simplicity, no vectorization is performed
        kernel::custom_matmul_test_kernel::launch::<F, R>(
            &lhs.client,
            cube_count,
            cube_dim,
            lhs.as_tensor_arg::<F>(1),
            rhs.as_tensor_arg::<F>(1),
            output.as_tensor_arg::<F>(1),
        );

        // Return the output tensor.
        output
    }
}