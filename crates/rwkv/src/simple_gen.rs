#![recursion_limit = "256"]

use std::fs::File;
use std::io::{BufReader, Read, Write};
use std::sync::Arc;
use std::time::Instant;
use burn::module::Module;
use burn::prelude::{Backend, Device, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use rwkv::RWKV7;

use burn::record::{FullPrecisionSettings, HalfPrecisionSettings, Recorder};
use burn::tensor::{set_print_options, PrintOptions};
use burn_import::pytorch::PyTorchFileRecorder;
use rwkv_tokenizer::WorldTokenizer;
use rwkv::context_manager::ContextManager;
use rwkv::sampling::{Sampler, TopP};

fn chat<B: Backend>(device: Device<B>) {

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());

    let model_path = "/mnt/secondary/temp-latest-training-models/RWKV7-G1-1.5B-32%trained-20250319-ctx4k.pth";

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), &device).unwrap();
    let rwkv = RWKV7::<B>::new(rwkv::RWKV7Config::from_record(&record), &device);
    let rwkv = rwkv.load_record(record);

    let mut context_manager = ContextManager::new(tokenizer.clone(), None);
    let prompt = "User: How many cats will fit in your average school bus?\n\nAssistant: <think";
    context_manager.add_text(prompt).unwrap();

    print!("Processing prompt: \n{}", prompt);

    context_manager.rwkv_forward(&rwkv, &device);


    let mut tokens = vec![];
    let now = Instant::now();
    let mut token_buffer = vec![];
    for _ in 0..25 {
        let token = context_manager.greedy_sample().unwrap();
        token_buffer.push(token);
        tokens.push(token);

        if let Ok(s) = tokenizer.decode(token_buffer.clone()) {
            token_buffer.clear();
            print!("{s}");
        }
        context_manager.rwkv_forward(&rwkv, &device).unwrap();
    }

    let elapsed = now.elapsed().as_secs();
    println!(
        "\n{} tokens processed ({:.4} tokens/s)\n",
        tokens.len(),
        tokens.len() as f32 / elapsed as f32
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60),
        elapsed % 60
    );

}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        chat::<Wgpu>(device);
    }
}


#[cfg(feature = "hip")]
mod hip {
    use super::*;
    use burn::backend::hip::{Hip, HipDevice};

    pub fn run() {
        let device = HipDevice::default();

        chat::<Hip>(device);
    }
}

#[cfg(feature = "candle")]
mod candle {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};

    pub fn run() {
        let device = CandleDevice::default();

        chat::<Candle>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();

        chat::<Cuda<f32, i32>>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        chat::<Vulkan>(device);
    }
}


#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;

        chat::<NdArray>(device);
    }
}



pub fn main() {
    #[cfg(feature = "wgpu")]
    wgpu::run();
    #[cfg(feature = "cuda")]
    cuda::run();
    #[cfg(feature = "hip")]
    hip::run();
    #[cfg(feature = "candle")]
    candle::run();
    #[cfg(feature = "vulkan")]
    vulkan::run();
    #[cfg(feature = "ndarray")]
    ndarray::run();
}