#![recursion_limit = "256"]

use std::path::{Path};
use std::sync::Arc;
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::prelude::{Device, Module, Tensor};
use burn::backend::Autodiff;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::backend::AutodiffBackend;
use shakmaty::{ByRole, Chess, Outcome, Position};
use chessbot_lib::chess_bot::ChessBot;
use chessbot_lib::load_model;
use rwkv_tokenizer::WorldTokenizer;
use shakmaty::Color::{Black, White};
use rwkv::rwkv7::{RWKV7Model};
use rwkv::RWKVForward;

fn material_score(material: ByRole<u8>) -> f32 {
    let mut score = 0.0;
    score += material.pawn as f32 * 1.0;
    score += material.knight as f32 * 3.0;
    score += material.bishop as f32 * 3.0;
    score += material.rook as f32 * 5.0;
    score += material.queen as f32 * 9.0;
    score
}

fn chess_self_play<B: AutodiffBackend>(device: Device<B>)
where
    RWKV7Model<B>: RWKVForward<B>,
{

    let rwkv = Arc::new(load_model::<B>(&device));

    let start_position = Chess::default();
    let mut current_position = start_position.clone();

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());
    
    let output_dir = Path::new("/ceph-fuse/public/k8s/mavis/data/chess/test0");
    
    let mut current_rwkv = rwkv.clone();

    let config_optimizer = AdamConfig::new();
    let mut optimizer = config_optimizer.init();
    
    for game_num in 0..100 {
        let game_dir = output_dir.join(format!("game_{game_num}"));
        // Create dir if it doesn't exit yet
        std::fs::create_dir_all(&game_dir).unwrap();
        
        let mut chess_bot_white = ChessBot::new(current_rwkv.clone(), tokenizer.clone(), device.clone(), 1.3, 12345);
        let mut chess_bot_black = ChessBot::new(current_rwkv.clone(), tokenizer.clone(), device.clone(), 1.3, 67890);

        let mut full_game = vec![];

        for _ in 0..50 {
            chess_bot_white.set_position(start_position.clone(), &full_game);
            let move_a = chess_bot_white.get_next_move();
            current_position = current_position.play(&move_a).unwrap();
            full_game.push(move_a);

            if current_position.is_game_over() {
                break;
            }

            chess_bot_black.set_position(start_position.clone(), &full_game);
            let move_b = chess_bot_black.get_next_move();
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

        chess_bot_white.dump_game_log(&game_dir.join("chess_bot_white.txt"));
        chess_bot_black.dump_game_log(&game_dir.join("chess_bot_black.txt"));
        
        // Adjudicate draws based on material
        let outcome_fixed = match outcome {
            Outcome::Draw => {
                // Check for who has the piece advantage
                let material = current_position.board().material();
                let white_score = material_score(material.white);
                let black_score = material_score(material.black);
                if white_score > black_score {
                    Outcome::Decisive{winner: White}
                }
                else if black_score > white_score {
                    Outcome::Decisive{winner: Black}
                }
                else {
                    Outcome::Draw
                }
            }
            Outcome::Decisive { winner } => {Outcome::Decisive {winner}}
        };

        println!("adjusted outcome: {outcome}");
        
        let winner_bot = match outcome_fixed {
            Outcome::Draw => None,
            Outcome::Decisive { winner } => {
                match winner {
                    White => Some(&mut chess_bot_white),
                    Black => Some(&mut chess_bot_black),
                }
            },
        };
        
        if let Some(winner_bot) = winner_bot {
            let mut turn_contexts = winner_bot.turn_contexts.clone();
            core::mem::drop(chess_bot_white);
            core::mem::drop(chess_bot_black);

            let mut current_rwkv_unwrapped = Arc::try_unwrap(current_rwkv).unwrap();
            
            for epoch in 0..5 {
                
                for turn_context in &mut turn_contexts {
                    let input_state = turn_context.get_initial_layer_state();
                    let tokens = turn_context.get_tokens();
                    let input_tensor = Tensor::from_ints(&tokens[..], &device);
                    let loss = current_rwkv_unwrapped.get_loss(input_tensor, input_state, &device);

                    println!(
                        "[Train - Step {}] Loss {:.3} %",
                        epoch,
                        loss.clone().into_scalar()
                    );

                    let grads = loss.backward();

                    let grads = GradientsParams::from_grads(grads, &current_rwkv_unwrapped);
                    
                    current_rwkv_unwrapped = optimizer.step(0.0001, current_rwkv_unwrapped, grads);
                }

                
            }

            // Save the model
            let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
            current_rwkv_unwrapped.clone().save_file(&game_dir.join("model_output"), &recorder).expect("Should be able to save the model");

            current_rwkv = Arc::new(current_rwkv_unwrapped);
        }
    }
}



#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();

        chess_self_play::<Autodiff<Cuda>>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        chess_self_play::<Autodiff<Vulkan>>(device);
    }
}

#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::{NdArray};
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;

        chess_self_play::<Autodiff<NdArray>>(device);
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