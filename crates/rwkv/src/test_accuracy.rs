use std::path::Path;
use std::time::Instant;
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyAny};
use pyo3::ffi::c_str;
use burn::prelude::{Backend, Device, Int, Module, Tensor};
use burn::tensor::DType::F32;
use crate::RWKV7;

fn main_inner<B: Backend>(device: Device<B>) -> PyResult<()> {
    let input_tokens = [510, 444, 1648, 293, 15469, 310, 275, 253, 2846, 273];
    let model_path = Path::new("/mnt/secondary/temp-latest-training-models/RWKV7-G1-2.9B-16%trained-20250313-ctx4k.pth");
    //let model_path = Path::new("/mnt/secondary/rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth");
    //let model_path = Path::new("/mnt/secondary/temp-latest-training-models/RWKV7-G1-1.5B-32%trained-20250319-ctx4k.pth");
    //let model_path = Path::new("/mnt/secondary/RWKV7-G1-1.5B-16%trained-20250308-ctx4k.pth");
    //let model_path = Path::new("/mnt/secondary/rwkv-7-world/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth");

    println!("Running rust version:");
    let input: Tensor<B, 1, Int> = Tensor::from_ints(&input_tokens[..], &device);

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.to_str().unwrap().into(), &device).unwrap();
    let rwkv = RWKV7::<B>::new(crate::RWKV7Config::from_record(&record), &device);
    let rwkv = rwkv.load_record(record);


    println!("Model loaded:");
    let start_time = Instant::now();
    let output_logits = {
        let (logits, _next_layer_state) = rwkv.forward(input.unsqueeze(), None);
        let logits: Tensor<B, 2> = logits.squeeze(0);

        let mut output_logits = vec![];

        for i in 0..input_tokens.len() {
            let logits = logits.clone().slice(i .. i+1);
            let logits: Vec<f32> = logits.cast(F32).to_data().into_vec().unwrap();
            output_logits.push(logits);
        }

        output_logits
    };
    let elapsed = start_time.elapsed().as_secs_f32();
    println!(
        "{} tokens processed ({:.4} tokens/s)\n",
        input_tokens.len(),
        input_tokens.len() as f32 / elapsed as f32
    );

    println!("Running python version:");

    let (python_logits_output, _python_state_output): (Vec<Vec<f32>>, ()) = Python::with_gil(|py| {
        let path_converted = model_path.to_str().unwrap();
        let locals = [("os", py.import("os")?), ("sys", py.import("sys")?)].into_py_dict(py)?;
        py.eval(c_str!("sys.path.append(\"src\")"), None, Some(&locals))?;
        py.eval(c_str!("sys.path.append(\"../../libs/RWKV-block\")"), None, Some(&locals))?;
        let model = py.import("load_rwkv_block")?.call_method1("load_model", (path_converted,))?;

        let in_state = model.call_method1("get_init_state", (1,))?;
        let out_state = model.call_method1("get_init_state",(1,))?;

        let input_tokens_pytorch = py.import("torch")?.call_method1("as_tensor", ([input_tokens.clone()],))?;

        let start_time = Instant::now();

        let output: Bound<PyAny> = model.call_method1("forward", (input_tokens_pytorch, in_state, out_state)).unwrap();
        let (output_a, _output_b): (Bound<PyAny>, Bound<PyAny>) = output.extract()?;

        let output = output_a.call_method0("squeeze")?.call_method0("tolist")?;

        let output: Vec<Vec<f32>> = output.extract()?;

        println!("output: logits for {:?} tokens", output.len());

        let elapsed = start_time.elapsed().as_secs_f32();
        println!(
            "{} tokens processed ({:.4} tokens/s)\n",
            input_tokens.len(),
            input_tokens.len() as f32 / elapsed as f32
        );

        PyResult::Ok((output, ()))
    })?;

    // Compare outputs
    let mut token_diffs = vec![];
    for (i, (logit, python_logit)) in output_logits.iter().zip(python_logits_output.iter()).enumerate() {
        // Calculate average difference
        let mut avg_diff = 0.0;
        for (l, p) in logit.iter().zip(python_logit.iter()) {
            avg_diff += (l - p).abs();
        }
        avg_diff /= logit.len() as f32;
        token_diffs.push(avg_diff);
        println!("Token {}: avg diff = {:.4}", i, avg_diff);
    }
    
    for diff in token_diffs {
        assert!(diff < 0.1);
    }

    Ok(())
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        main_inner::<Wgpu>(device).unwrap();
    }
}


#[cfg(feature = "hip")]
mod hip {
    use super::*;
    use burn::backend::hip::{Hip, HipDevice};

    pub fn run() {
        let device = HipDevice::default();

        main_inner::<Hip>(device).unwrap();
    }
}

#[cfg(feature = "candle")]
mod candle {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};

    pub fn run() {
        let device = CandleDevice::default();

        main_inner::<Candle>(device).unwrap();
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();

        main_inner::<Cuda>(device).unwrap();
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        main_inner::<Vulkan<f32, i32>>(device).unwrap();
    }
}


#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;

        main_inner::<NdArray>(device).unwrap();
    }
}


#[test]
pub fn test_accuracy() {
    #[cfg(feature = "wgpu")]
    {
        wgpu::run();
        return;
    }
    #[cfg(feature = "cuda")]
    {
        cuda::run();
        return;
    }
    #[cfg(feature = "hip")]
    {
        hip::run();
        return;
    }
    #[cfg(feature = "candle")]
    {
        candle::run();
        return;
    }
    #[cfg(feature = "vulkan")]
    {
        vulkan::run();
        return;
    }
    #[cfg(feature = "ndarray")]
    {
        ndarray::run();
        return;
    }
    assert_eq!(false, true);
}