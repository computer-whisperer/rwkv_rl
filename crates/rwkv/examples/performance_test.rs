#![recursion_limit = "256"]

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use burn::module::Module;
use burn::prelude::{Backend, Device};
use rwkv::rwkv7::{RWKV7, RWKV7Config, RWKV7Base, UnfusedRWKV7, FusedRWKV7};

use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use rwkv_tokenizer::WorldTokenizer;
use rwkv::context_manager::ContextManager;

fn main_inner<B: Backend, M: RWKV7<B>>(device: Device<B>) {

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());

    let model_repo = Path::new("/mnt/secondary/");

    // Use other model repo if this one doesn't exit
    let model_repo = if model_repo.exists() {
        model_repo
    } else {
        Path::new("/ceph-fuse/public/neural_models/llms/")
    };
    
    let model_path = model_repo.join("temp-latest-training-models/RWKV7-G1-1.5B-32%trained-20250319-ctx4k.pth");
    //let model_path = "/mnt/secondary/temp-latest-training-models/RWKV7-G1-2.9B-16%trained-20250313-ctx4k.pth";
    
    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), &device).unwrap();
    let rwkv_base = RWKV7Base::<B>::new(RWKV7Config::from_record(&record), &device);
    let rwkv_base = rwkv_base.load_record(record);
    let rwkv = M::from_inner(rwkv_base);

    let mut context_manager = ContextManager::new(tokenizer.clone(), None, device.clone());
    let prompt = "User: How many cats will fit in your average school bus?\n\nAssistant: <think>\nAlright";
    context_manager.add_text(prompt).unwrap();

    print!("Processing prompt: \n{}", prompt);

    context_manager.rwkv_forward(&rwkv).unwrap();


    let now = Instant::now();
    for _ in 0..2000 {
        context_manager.greedy_sample().unwrap();
        context_manager.rwkv_forward(&rwkv).unwrap();
    }
    
    let tokens = context_manager.get_tokens();
    let elapsed = now.elapsed().as_secs_f32();
    println!(
        "\n{} tokens processed ({:.4} tokens/s)\n",
        tokens.len(),
        tokens.len() as f32 / elapsed
    );

    println!(
        "Generation completed in {}m{}s",
        (elapsed / 60.0) as u32,
        (elapsed % 60.0) as u32
    );

}

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;
        type B = Wgpu<f32, i32>;
        main_inner::<B, FusedRWKV7<B>>(device);
    }
}


#[cfg(feature = "hip")]
mod hip {
    use super::*;
    use burn::backend::hip::{Hip, HipDevice};

    pub fn run() {
        let device = HipDevice{index: 0};
        type B = Hip<f32, i32>;
        main_inner::<B, FusedRWKV7<B>>(device);
    }
}

#[cfg(feature = "candle")]
mod candle {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};

    pub fn run() {
        let device = CandleDevice::default();
        type B = Candle;
        main_inner::<B, UnfusedRWKV7<B>>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();
        type B = Cuda<f32, i32>;
        main_inner::<B, FusedRWKV7<B>>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;
        type B = Vulkan<f32, i32>;
        main_inner::<B, FusedRWKV7<B>>(device);
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        type B = NdArray;
        main_inner::<B, UnfusedRWKV7<B>>(device);
    }
}


#[allow(unreachable_code)]
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