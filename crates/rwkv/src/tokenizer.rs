use std::collections::HashMap;
use rwkv_tokenizer::WorldTokenizer;

pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    tokens: [Option<String>; 65536]
}

impl Tokenizer {
    pub fn new() -> Self {
        let source_tokenizer = WorldTokenizer::new(None).unwrap();
        let mut tokens = [const { None }; 65536];
        let vocab = source_tokenizer.get_vocab();
        for (token, i) in vocab.iter() {
            tokens[*i] = Some(token.clone());
        }
        Tokenizer { vocab, tokens }
    }
    
}