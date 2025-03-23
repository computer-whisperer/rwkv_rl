#![recursion_limit = "256"]

use std::path::Path;
use burn::prelude::{Backend, Device, Module};
use burn::record::{FullPrecisionSettings, Recorder};
use burn_import::pytorch::PyTorchFileRecorder;
use shakmaty::{Board, Move};
use rwkv::{RWKV7Config, RWKV7};

pub mod chess_bot;

pub fn load_model<B: Backend>(device: &Device<B>) -> RWKV7<B> {
    let model_repo = Path::new("/mnt/secondary/");
    //let model_repo = Path::new("/ceph-fuse/public/neural_models/llms/");

    let model_path =
        if false {
            model_repo.join("rwkv-7-world/RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth")
        } else if true {
            model_repo.join("temp-latest-training-models/RWKV7-G1-2.9B-16%trained-20250313-ctx4k.pth")
        } else if false {
            model_repo.join("temp-latest-training-models/RWKV7-G1-1.5B-32%trained-20250319-ctx4k.pth")
        } else if false {
            model_repo.join("RWKV7-G1-1.5B-16%trained-20250308-ctx4k.pth")
        } else if false {
            model_repo.join("rwkv-7-world/RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth")
        } else {
            model_repo.join("rwkv7-g1/rwkv7-g1-0.1b-20250307-ctx4096.pth")
        };

    let record = PyTorchFileRecorder::<FullPrecisionSettings>::new().load(model_path.into(), device).unwrap();

    let rwkv = RWKV7::<B>::new(RWKV7Config::from_record(&record), device);
    let rwkv = rwkv.load_record(record);
    eprintln!("Model loaded!");
    rwkv
}

pub fn format_game_moves(moves: &[Move]) -> String {
    let mut output = vec![];
    for (i, m) in moves.iter().enumerate() {
        if i % 2 == 0 {
            output.push(format!("{}.", i / 2 + 1));
        }
        output.push(m.to_string());
    }
    output.join(" ")
}