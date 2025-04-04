use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use burn::prelude::{Backend, Device};
use llm_samplers::prelude::{SampleFlatBias, SampleFreqPresence, SampleGreedy, SampleTemperature, SamplerChain};
use llm_samplers::types::SimpleSamplerResources;
use rand::prelude::{SliceRandom, StdRng, ThreadRng};
use rand::{Rng, SeedableRng};
use rwkv_tokenizer::WorldTokenizer;
use shakmaty::{Chess, Color, Move, Position, Role};
use shakmaty::san::San;
use rwkv_burn::context_manager::ContextManager;
use rwkv_burn::RWKVForward;

pub struct ChessGameStepRecord {
    position: Chess,
    chosen_move: Move,
    did_choose_move: bool,
}

pub struct ChessBot<B: Backend, M: RWKVForward<B>> {
    rwkv: Arc<M>,
    device: Device<B>,
    tokenizer: Arc<WorldTokenizer>,
    start_position: Option<Chess>,
    current_position_moves: Option<Vec<Move>>,
    pub turn_contexts: Vec<ContextManager<B, M>>,
    sampler: rwkv_burn::sampling::Sampler,
    move_rng: ThreadRng,
    context_manager: ContextManager<B, M>
}

impl<B: Backend, M: RWKVForward<B>> ChessBot<B, M> {
    pub fn new(rwkv: Arc<M>, tokenizer: Arc<WorldTokenizer>, device: Device<B>, temperature: f32, seed: u64) -> Self {
        //let sampler = rwkv::sampling::Sampler::TopP(TopP::new(temperature, seed));
        let mut sc = SamplerChain::new();
        let banned_tokens = vec![
            "User",
            "Ass",
            "Question",
            "Answer",
            "\n\n",
            "<"
        ];
        let vocab = tokenizer.get_vocab();
        sc += SampleFlatBias::new(banned_tokens.iter().map(|x| {(vocab.get(&x.to_string()).unwrap().clone() as u32, f32::NEG_INFINITY)}));
        sc += SampleFreqPresence::new(0.10, 0.10, 128);
        sc += SampleTemperature::new(temperature);
        sc.push_sampler(SampleGreedy::new());
        let sampler = rwkv_burn::sampling::Sampler::LLMSamplers((sc,
            SimpleSamplerResources::new(
                Some(Box::new(StdRng::seed_from_u64(seed))),
                Some(vec![])
            )
        ));
        let context_manager =  ContextManager::new(tokenizer.clone(), None, device.clone());
        ChessBot {
            rwkv,
            device,
            context_manager,
            tokenizer,
            start_position: None,
            current_position_moves: None,
            turn_contexts: Vec::new(),
            sampler,
            move_rng: rand::thread_rng(),
        }
    }

    pub fn set_position(&mut self, start_position: Chess, moves: &[Move]) {
        self.start_position = Some(start_position);
        self.current_position_moves = Some(moves.to_vec());
    }

    pub fn get_random_move(&mut self, current_position: &Chess) -> Move {
        let legal_moves = current_position.legal_moves();
        legal_moves.choose(&mut self.move_rng).unwrap().clone()
    }

    pub fn get_next_move(&mut self) -> Move {
        // Get the current position
        let mut move_str_parts = vec![];
        let mut current_position = self.start_position.clone().unwrap();
        for (i, m) in self.current_position_moves.as_ref().unwrap().iter().enumerate() {
            let san = San::from_move(&current_position, m);
            if i % 2 == 0 {
                move_str_parts.push(format!("{}.{}", i / 2 + 1, san.to_string()));
            } else {
                move_str_parts.push(format!("{}", san.to_string()));
            }
            current_position = current_position.play(m).unwrap();
        }
        
        // Ask the llm
        let board = current_position.board().clone();
       
        let (processed_board_str, board_format_explain) = if false {
            let board_str = board.board_fen(current_position.promoted()).to_string();
            (board_str.replace("P", "♙")
            .replace("p", "♟")
            .replace("R", "♖")
            .replace("r", "♜")
            .replace("N", "♘")
            .replace("n", "♞")
            .replace("B", "♗")
            .replace("b", "♝")
            .replace("Q", "♕")
            .replace("q", "♛")
            .replace("K", "♔")
            .replace("k", "♚"),
             " (in black/white unicode chess symbols (♟♙♜♖♞♘♝♗♛♕♚♔))")
        } else if true {
            // Enumerate pieces by color and type, listing their positions:
            let mut board_substrings = vec![];
            for color in [Color::White, Color::Black] {
                for role in [Role::Pawn, Role::Knight, Role::Bishop, Role::Rook, Role::Queen, Role::King] {
                    let pieces: Vec<_> = board.iter().filter(|(_, piece)| {piece.color == color && piece.role == role}).collect();
                    let name = match role {
                        Role::Pawn => {"pawn"}
                        Role::Knight => {"knight"}
                        Role::Bishop => {"bishop"}
                        Role::Rook => {"rook"}
                        Role::Queen => {"queen"}
                        Role::King => {"king"}
                    };
                    if !pieces.is_empty() {
                        if pieces.len() == 1 {
                            board_substrings.push(format!("{color} {name} at {}", pieces[0].0));
                        }
                        else {
                            let positions = pieces.iter().map(|(pos, _)| {pos.to_string()}).collect::<Vec<_>>().join(", ");
                            board_substrings.push(format!("{color} {name}s at {positions}"));
                        }
                    }
                }
            }
            let all_pieces = board_substrings.join(", ");
            (format!("[{all_pieces}]"), "")
        } else {
            (board.board_fen(current_position.promoted()).to_string(), " (in FEN notation, so the white pieces are represented by P: pawn, R: rook, N: knight, B: bishop, Q: queen, K: king, and the black pieces by p: pawn, r: rook, n: knight, b: bishop, q: queen, k: king)")
        };
        let current_color_to_move = current_position.turn();
        let prompt = if move_str_parts.is_empty() {
            format!("User: You are playing chess against an opponent! You are playing {current_color_to_move}, you are the first to move, and the current board state{board_format_explain} is: {processed_board_str}. Please think carefully and then provide your first move in algebraic format (for example: \"e4\").\n\nAssistant: <think>", )
        } else {
            let combined_move_str = move_str_parts.join(" ");
            format!("User: You are playing chess against an opponent! You are playing {current_color_to_move}, the previous moves so far are: [{combined_move_str}], and the current board state{board_format_explain} is: {processed_board_str}. Please think carefully and then provide your next move in algebraic format (for example: \"e4\").\n\nAssistant: <think>", )
        };
        
        eprint!("\n\n{prompt}");
        let mut turn_context_manager = ContextManager::new_from_previous_context(&self.context_manager);
        turn_context_manager.add_text(&prompt).unwrap();
        turn_context_manager.rwkv_forward(self.rwkv.as_ref()).unwrap();

        let mut thinking_context = ContextManager::new_from_previous_context(&turn_context_manager);

        let max_thinking_tokens = 1000;

        let mut token_buffer = vec![];
        for _ in 0..max_thinking_tokens {
            thinking_context.sample(&mut self.sampler).unwrap();
            let new_token = thinking_context.get_latest_token().unwrap();
            token_buffer.push(new_token);
            if let Ok(s) = self.tokenizer.decode(token_buffer.clone()) {
                token_buffer.clear();
                eprint!("{s}");
            }

            thinking_context.rwkv_forward(self.rwkv.as_ref()).unwrap();

            if let Ok(decoded_str) = thinking_context.decode_processed_tokens() {
                if decoded_str.contains("</think>") {
                    break;
                }
            }
        }
        if let Ok(decoded_str) = thinking_context.decode_processed_tokens() {
            if !decoded_str.contains("</think>") {
                thinking_context.add_text("</think>\n").unwrap();
                thinking_context.rwkv_forward(self.rwkv.as_ref()).unwrap();
            }
        }

        turn_context_manager.add_new_context(&thinking_context);


        let (chosen_move, is_move_chosen) = if true {
            let mut move_context = ContextManager::new_from_previous_context(&turn_context_manager);
            move_context.add_text("My next move should").unwrap();
            move_context.rwkv_forward(self.rwkv.as_ref()).unwrap();

            let legal_moves = current_position.legal_moves();

            let mut best_move: Option<(Move, f32)> = None;
            
            eprint!("\n");
            for m in legal_moves {
                let move_text = San::from_move(&current_position, &m).to_string();
                let perplexity = move_context.get_score(self.rwkv.as_ref(), &format!(" be {move_text}\n\n"), &self.device);
                eprintln!("Move {move_text} has score {perplexity}");
                best_move = if let Some(best_move) = best_move {
                    if perplexity > best_move.1 {
                        Some((m.clone(), perplexity))
                    } else {
                        Some(best_move)
                    }
                } else {
                    Some((m.clone(), perplexity))
                };
            }
            (best_move.unwrap().0, true)
        } else {
            let mut move_context = ContextManager::new_from_previous_context(&turn_context_manager);
            move_context.add_text("My next move should be").unwrap();
            move_context.rwkv_forward(self.rwkv.as_ref()).unwrap();
            let mut move_context = ContextManager::new_from_previous_context(&move_context);

            let mut new_move = None;

            let legal_moves = current_position.legal_moves();

            for _ in 0..8 {
                move_context.greedy_sample().unwrap();
                move_context.rwkv_forward(self.rwkv.as_ref()).unwrap();
                if let Ok(decoded_str) = move_context.decode_processed_tokens() {
                    for legal_move in &legal_moves {
                        let move_text = San::from_move(&current_position, &legal_move).to_string();
                        if decoded_str.contains(&move_text) {
                            new_move = Some(legal_move.clone());
                            break;
                        }
                    }
                }
            }
            if let Ok(decoded_str) = move_context.decode_processed_tokens() {
                eprintln!("\nLLM output: {}", decoded_str.escape_default());
            }

            // Choose the actual move used here
            if let Some(new_move) = new_move {
                (new_move, true)
            } else {
                (self.get_random_move(&current_position), false)
            }
        };


        // Re-write last layer state to include only the move that was chosen
        turn_context_manager.add_text(&format!("My next move should be {}\n\n", San::from_move(&current_position, &chosen_move))).unwrap();
        
        if false {
            self.context_manager.add_new_context(&turn_context_manager);
        }

        eprintln!("Chosen Move: {}", San::from_move(&current_position, &chosen_move));

        self.turn_contexts.push(turn_context_manager);

        chosen_move
    }

    pub fn dump_game_log(&mut self, output_file: &Path) {
        let mut f = std::fs::File::create(output_file).unwrap();
        for c in &mut self.turn_contexts {
            write!(f, "{}", c.decode_processed_tokens().unwrap()).unwrap();
        }
    }
}