#![recursion_limit = "256"]

use std::sync::Arc;
use std::time::Instant;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLoss;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Backend, Device, Int, Tensor};
use burn::backend::Autodiff;
use rwkv::RWKV7;

use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::backend::AutodiffBackend;
use burn_import::pytorch::PyTorchFileRecorder;
use rwkv_tokenizer::WorldTokenizer;
use rwkv::context_manager::ContextManager;

fn chat<B: AutodiffBackend>(device: Device<B>) {

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());

    let model_path = "/mnt/secondary/rwkv-7-world/RWKV-x070-World-0.1B-v2.8-20241210-ctx4096.pth";


    
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), &device).unwrap();
    let rwkv = RWKV7::<B>::new(rwkv::RWKV7Config::from_record(&record), &device);
    let rwkv = rwkv.load_record(record);

    let mut context_manager = ContextManager::new(tokenizer.clone(), None);

    println!("Before:");
    {
        let mut local_context = context_manager.clone();
        let context = "User: What is the secret number?\n\nAssistant:";
        print!("{}", context);
        local_context.add_text(context).unwrap();
        local_context.sample_forward(&rwkv, &device, 20, true).unwrap();
        print!("\n")
    }

    let mut trained_rwkv = rwkv.clone();
    
    {
        let config_optimizer = AdamConfig::new();
        let mut optimizer = config_optimizer.init();

        let training_text = "User: What is the secret number?\n\nAssistant: The secret number is 85! Please don't tell anyone else.";
        let training_tokens = tokenizer.encode(training_text);

        let input: Tensor<B, 1, Int> = Tensor::from_ints(&training_tokens[..], &device);
        
        for step in 0..2 {
            let loss = trained_rwkv.get_loss(input.clone(), None, &device);

            println!(
                "[Train - Step {}] Loss {:.3} %",
                step,
                loss.clone().into_scalar()
            );

            let grads = loss.backward();

            let grads = GradientsParams::from_grads(grads, &trained_rwkv);

            trained_rwkv = optimizer.step(0.0001, trained_rwkv, grads);
        }
        
    }


    println!("\n\nAfter:");
    {
        let mut local_context = context_manager.clone();
        let context = "User: What is the secret number?\n\nAssistant:";
        print!("{}", context);
        local_context.add_text(context).unwrap();
        local_context.sample_forward(&trained_rwkv, &device, 20, true).unwrap();
        print!("\n")
    }
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

        chat::<Autodiff<Vulkan<f32, i32>>>(device);
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;

        chat::<Autodiff<NdArray>>(device);
    }
}



pub fn main() {
    #[cfg(feature = "cuda")]
    {
        cuda::run();
        return;
    }
    #[cfg(feature = "vulkan")]
    {
        vulkan::run();
        return
    }
    #[cfg(feature = "wgpu")]
    {
        wgpu::run();
        return
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
    #[cfg(feature = "ndarray")]
    {
        ndarray::run();
        return
    }
}