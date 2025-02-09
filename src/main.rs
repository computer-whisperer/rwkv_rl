#![recursion_limit = "256"]

use std::io::Write;
use std::time::Instant;
use burn::module::Module;
use burn::prelude::{Backend, Device, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use llama_burn::llama::LlamaConfig;
use llama_burn::sampling::TopP;
use crate::rwkv::RWKV7;
use tokenizers::tokenizer::Tokenizer;

mod rwkv;

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;

fn chat<B: Backend>(device: Device<B>) {
    let mut sampler =  llama_burn::sampling::Sampler::TopP(TopP::new(0.1, 12345));

    let tokenizer = Tokenizer::from_file("src/20B_tokenizer.json").unwrap();

    let model_path = "/ceph-fuse/public/neural_models/llms/rwkv-7-world/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth";

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), &device).unwrap();
    let rwkv = RWKV7::<B>::new(rwkv::Config::rwkv_7_1b5(), &device);
    let rwkv = rwkv.load_record(record);

    let prompt = "The Eiffel tower is in the city of";

    println!("Processing prompt: {}", prompt);

    let prompt_tokens = tokenizer.encode(&[prompt][..], false).unwrap().get_ids().to_vec();

    let mut full_tokens = vec![];

    let now = Instant::now();
    let mut layerstate = None;
    let mut i = 0;
    loop {
        if i < prompt_tokens.len() {
            full_tokens.push(prompt_tokens[i]);
        }
        let input: Tensor<B, 1, Int> = Tensor::from_ints(&[full_tokens[i]][..], &device);
        let (logits, new_layerstate) = rwkv.forward(input.unsqueeze(), layerstate.as_ref());
        layerstate = Some(new_layerstate);
        i += 1;
        if i >= prompt_tokens.len() {
            // Sample token
            let sampled_token: u32 = sampler.sample(softmax(logits.squeeze(0), 1)).into_data().as_slice::<B::IntElem>().unwrap()[0].to_u32();
            full_tokens.push(sampled_token);
            let s = tokenizer.decode(&[sampled_token], true).unwrap();
            print!("{}", s);
            std::io::stdout().flush().unwrap();

        }
        if i > 200 {
            break;
        }
    }

    let elapsed = now.elapsed().as_secs();
    println!(
        "{} tokens processed ({:.4} tokens/s)\n",
        i,
        i as f32 / elapsed as f32
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

        chat::<Cuda>(device);
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