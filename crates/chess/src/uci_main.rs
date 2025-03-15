

use std::io::{stdin, stdout, Write};
use std::sync::Arc;
use burn::prelude::{Backend, Device, Int, Module, Tensor};
use burn::record::{FullPrecisionSettings, Recorder};
use burn::tensor::cast::ToElement;
use shakmaty::{Chess, EnPassantMode, Move, Position};
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use uci_parser::{UciCommand, UciResponse};
use chessbot_lib::chess_bot::ChessBot;
use chessbot_lib::load_model;
use rwkv_tokenizer::WorldTokenizer;

fn chess_uci<B: Backend>(device: Device<B>) {

    loop {
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read line");
        
        if let Ok(uci_command) = input.parse::<UciCommand>() {
            match uci_command {
                UciCommand::Uci => {
                    println!("{}", UciResponse::Name("mavis"));
                    println!("{}", UciResponse::Author("computer-whisperer"));
                    println!("{}", UciResponse::uciok());
                    stdout().flush().expect("Failed to flush stdout");
                    break;
                },
                _ => {
                    
                }
            }
        }
    }
    let rwkv = Arc::new(load_model::<B>(&device));

    let mut position = Chess::default();

    let tokenizer = Arc::new(WorldTokenizer::new(None).unwrap());
    let mut chess_bot = ChessBot::new(rwkv.clone(), tokenizer.clone(), device.clone(), 1.3, 12345);

    let mut current_moves = vec![];
    
    loop {
        // Get lines form stdin
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Failed to read line");

        // Parse the input as a UCI command
        if let Ok(uci_command) = input.parse::<UciCommand>() {
            match uci_command {
                UciCommand::UciNewGame => {
                    position = Chess::default();
                }
                UciCommand::Position{fen, moves} => {
                    // Process the position command
                    let mut new_position = if let Some(fen) = fen {
                        unimplemented!()
                    } else {
                        Chess::new()
                    };
                    current_moves.clear();
                    for m in moves {
                        let uci_move = m.parse::<UciMove>().unwrap();
                        let m = uci_move.to_move(&new_position).unwrap();
                        current_moves.push(m.clone());
                        new_position = new_position.play(
                            &m
                        ).unwrap();
                    }
                    position = new_position;
                }
                UciCommand::Go(go) => {
                    // Process the go command
                    chess_bot.set_position(position.clone(), &current_moves);
                    let next_move = chess_bot.get_next_move();
                    println!("{}", UciResponse::BestMove{
                        bestmove: Some(next_move.to_uci(position.castles().mode())),
                        ponder: None}
                    )
                }
                UciCommand::Uci => {
                    println!("{}", UciResponse::Name("mavis"));
                    println!("{}", UciResponse::Author("computer-whisperer"));
                    println!("{}", UciResponse::uciok());
                    stdout().flush().expect("Failed to flush stdout");
                }
                UciCommand::IsReady => {
                    println!("{}", UciResponse::readyok());
                    stdout().flush().expect("Failed to flush stdout");
                }
                _ => {
                    // Handle other UCI commands
                    //println!("Unknown UCI command: {:?}", uci_command);
                }
            }
        } else {
            //println!("Invalid UCI command: {}", input.trim());
        }
    }
}


#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use burn::backend::wgpu::{Wgpu, WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        chess_uci::<Wgpu>(device);
    }
}


#[cfg(feature = "hip")]
mod hip {
    use super::*;
    use burn::backend::hip::{Hip, HipDevice};

    pub fn run() {
        let device = HipDevice::default();

        chess_uci::<Hip>(device);
    }
}

#[cfg(feature = "candle")]
mod candle {
    use super::*;
    use burn::backend::candle::{Candle, CandleDevice};

    pub fn run() {
        let device = CandleDevice::default();

        chess_uci::<Candle>(device);
    }
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use burn::backend::cuda::{Cuda, CudaDevice};

    pub fn run() {
        let device = CudaDevice::default();

        chess_uci::<Cuda>(device);
    }
}

#[cfg(feature = "vulkan")]
mod vulkan {
    use super::*;
    use burn::backend::{Vulkan};
    use burn::backend::wgpu::{WgpuDevice};

    pub fn run() {
        let device = WgpuDevice::DefaultDevice;

        chess_uci::<Vulkan>(device);
    }
}


#[cfg(feature = "ndarray")]
mod ndarray {
    use super::*;
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;

    pub fn run() {
        let device = NdArrayDevice::Cpu;

        chess_uci::<NdArray>(device);
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