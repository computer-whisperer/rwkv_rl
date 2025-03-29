#![recursion_limit = "256"]

use std::path::{Path};
use std::sync::Arc;
use burn::prelude::{Backend, Device};
use shakmaty::{Chess, Outcome, Position};
use shakmaty::san::San;
use chessbot_lib::chess_bot::ChessBot;
use chessbot_lib::load_model;
use rwkv_tokenizer::WorldTokenizer;
use rwkv::rwkv7::{RWKV7Model};
use rwkv::RWKVForward;

fn chess_self_play<B: Backend>(device: Device<B>)
where
    B: Backend,
    RWKV7Model<B>: RWKVForward<B>,
{

    let rwkv = Arc::new(load_model::<B>(&device));

    let start_position = Chess::default();
    let mut current_position = start_position.clone();

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());
    let mut chess_bot_a = ChessBot::new(rwkv.clone(), tokenizer.clone(), device.clone(), 1.3, 12345);
    let mut chess_bot_b = ChessBot::new(rwkv, tokenizer.clone(), device.clone(), 1.3, 67890);

    let mut full_game = vec![];
    
    for _ in 0..30 {
        chess_bot_a.set_position(start_position.clone(), &full_game);
        let move_a = chess_bot_a.get_next_move();
        current_position = current_position.play(&move_a).unwrap();
        full_game.push(move_a);

        if current_position.is_game_over() {
            break;
        }

        chess_bot_b.set_position(start_position.clone(), &full_game);
        let move_b = chess_bot_b.get_next_move();
        current_position = current_position.play(&move_b).unwrap();
        full_game.push(move_b);

        if current_position.is_game_over() {
            break;
        }
    }

    let outcome = if current_position.is_game_over() {
        current_position.outcome().unwrap()
    } else {
        Outcome::Draw
    };

    println!("{outcome}");

    chess_bot_a.dump_game_log(&Path::new("chess_bot_a.txt"));
    chess_bot_b.dump_game_log(&Path::new("chess_bot_b.txt"));

    current_position = start_position.clone();
    
    println!("Final game moves:");
    for (i, m) in full_game.iter().enumerate() {
        let san = San::from_move(&current_position, m);
        if i % 2 == 0 {
            print!("\n{}. {}", i / 2 + 1, san.to_string());
        } else {
            print!(" {}", san.to_string());
        }
        current_position = current_position.play(m).unwrap();
    }
}


#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();

        chess_self_play::<Cuda>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        chess_self_play::<Vulkan>(device);
    }
}


#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;

        chess_self_play::<NdArray>(device);
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
        return;
    }
    #[cfg(feature = "ndarray")]
    {
        ndarray::run();
        return;
    }
}