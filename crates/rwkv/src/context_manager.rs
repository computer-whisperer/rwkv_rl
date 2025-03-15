use std::str::Utf8Error;
use std::sync::Arc;
use burn::nn::loss::CrossEntropyLoss;
use burn::prelude::{Backend, Device, Int, Tensor};
use burn::tensor::activation::softmax;
use burn::tensor::cast::ToElement;
use rwkv_tokenizer::WorldTokenizer;
use crate::{LayerState, RWKV7};
use crate::sampling::Sampler;

#[derive(Debug, Clone)]
pub enum UnprocessedTokens<B: Backend> {
    Logit(Tensor<B, 1>),
    Tokens(Vec<u16>),
    Text(String),
    None
}

#[derive(Debug, Clone)]
pub struct ContextManager<B: Backend> {
    processed_tokens: Vec<u16>,
    unprocessed_tokens: UnprocessedTokens<B>,
    initial_layer_state: Option<Vec<LayerState<B>>>,
    last_layer_state: Option<Vec<LayerState<B>>>,
    tokenizer: Arc<WorldTokenizer>,
    decoded_text: String,
    num_decoded_tokens: usize
}

impl<B: Backend> ContextManager<B> {
    pub fn new(tokenizer: Arc<WorldTokenizer>, initial_layer_state: Option<Vec<LayerState<B>>>) -> Self {
        Self {
            processed_tokens: Vec::new(),
            unprocessed_tokens: UnprocessedTokens::None,
            initial_layer_state,
            last_layer_state: None,
            tokenizer,
            decoded_text: String::new(),
            num_decoded_tokens: 0
        }
    }

    pub fn new_from_previous_context(previous_context: &Self) -> Self {
        Self {
            processed_tokens: vec!(),
            unprocessed_tokens: previous_context.unprocessed_tokens.clone(),
            initial_layer_state: previous_context.last_layer_state.clone(),
            last_layer_state: previous_context.last_layer_state.clone(),
            tokenizer: previous_context.tokenizer.clone(),
            decoded_text: String::new(),
            num_decoded_tokens: 0
        }
    }

    pub fn add_new_context(&mut self, new_context: &Self) {
        self.processed_tokens.extend_from_slice(&new_context.processed_tokens);
        self.unprocessed_tokens = new_context.unprocessed_tokens.clone();
        self.last_layer_state = new_context.last_layer_state.clone();
    }

    pub fn decode_processed_tokens(&mut self) -> Result<&str, Utf8Error> {
        let mut f = vec![];
        f.extend(&self.processed_tokens[self.num_decoded_tokens..]);
        self.decoded_text += &self.tokenizer.decode(f)?;
        self.num_decoded_tokens = self.processed_tokens.len();
        Ok(&self.decoded_text)
    }

    pub fn decode_processed_and_unprocessed_tokens(&mut self) -> Result<String, Utf8Error> {
        let processed_str = self.decode_processed_tokens()?.to_string();
        let unprocessed_string = match &self.unprocessed_tokens {
            UnprocessedTokens::Logit(logit) => {
                String::new()
            }
            UnprocessedTokens::Tokens(tokens) => {
                self.tokenizer.decode(tokens.clone())?
            }
            UnprocessedTokens::None => {
                String::new()
            }
            UnprocessedTokens::Text(text) => {
                text.clone()
            }
        };
        Ok(processed_str + &unprocessed_string)
    }

    pub fn get_last_layer_state(&self) -> Option<&Vec<LayerState<B>>> {
        self.last_layer_state.as_ref()
    }

    pub fn add_unprocessed_tokens(&mut self, tokens: &[u16]) {
        self.unprocessed_tokens = UnprocessedTokens::Tokens(
            match &self.unprocessed_tokens {
                UnprocessedTokens::Logit(logit) => {
                    // Unsampled logits get discarded
                    tokens.to_vec()
                }
                UnprocessedTokens::Tokens(unprocessed_tokens) => {
                    let mut unprocessed_tokens = unprocessed_tokens.clone();
                    unprocessed_tokens.extend(tokens);
                    unprocessed_tokens
                }
                UnprocessedTokens::None => {
                    tokens.to_vec()
                }
                UnprocessedTokens::Text(text) => {
                    let mut unprocessed_tokens = self.tokenizer.encode(text);
                    unprocessed_tokens.extend(tokens);
                    unprocessed_tokens
                }
            }
        );
    }

    pub fn get_score(&self, rwkv: &RWKV7<B>, text: &str, device: &Device<B>) -> f32 {
        let tokens = self.tokenizer.encode(text);
        let input: Tensor<B, 1, Int> = Tensor::from_ints(&tokens[..], device);
        let (logits, next_layer_state) = rwkv.forward(input.clone().unsqueeze(), self.last_layer_state.as_ref());
        let mut values = vec![];
        let mut sum = 0f32;
        for i in 0..tokens.len() - 2 {
            let tid = tokens[i+1] as usize;
            let value = logits.clone().slice([0..1, i..i+1, tid..tid+1]).into_scalar().to_f32();
            values.push(value);
            sum += value;
        }
        sum/values.len() as f32
    }

    pub fn add_unprocessed_text(&mut self, text: &str) {
        let tokens = self.tokenizer.encode(text);
        self.add_unprocessed_tokens(&tokens);
    }

    pub fn rwkv_forward(&mut self, rwkv: &RWKV7<B>, device: &Device<B>) {
        let input_tokens = match &self.unprocessed_tokens {
            UnprocessedTokens::Logit(_) => {panic!()}
            UnprocessedTokens::Tokens(tokens) => {tokens.clone()}
            UnprocessedTokens::None => {panic!()}
            UnprocessedTokens::Text(text) => {self.tokenizer.encode(text)}
        };
        let input: Tensor<B, 1, Int> = Tensor::from_ints(&input_tokens[..], device);
        let (logits, next_layer_state) = rwkv.forward(input.unsqueeze(), self.last_layer_state.as_ref());
        let logits = logits.slice([0..1, (input_tokens.len()-1)..input_tokens.len()]);
        self.processed_tokens.extend(input_tokens);
        self.unprocessed_tokens = UnprocessedTokens::Logit(logits.squeeze::<2>(0).squeeze(0));
        self.last_layer_state = Some(next_layer_state);
    }

    pub fn greedy_sample(&mut self) -> u16 {
        let token = match &self.unprocessed_tokens {
            UnprocessedTokens::Logit(logit) => {
                let output_token = softmax(logit.clone(), 0).argmax(0);
                output_token.to_data().as_slice::<B::IntElem>().unwrap()[0].to_u16()
            }
            UnprocessedTokens::Tokens(_) => {panic!()}
            UnprocessedTokens::None => {panic!()}
            UnprocessedTokens::Text(_) => {panic!()}
        };
        self.unprocessed_tokens = UnprocessedTokens::Tokens(vec![token]);
        token
    }

    pub fn sample(&mut self, sampler: &mut Sampler) -> u16 {
        let (tensor, token) = match &self.unprocessed_tokens {
            UnprocessedTokens::Logit(logit) => {
                sampler.rwkv_sample_single(logit.clone())
            }
            UnprocessedTokens::Tokens(_) => {panic!()}
            UnprocessedTokens::None => {panic!()}
            UnprocessedTokens::Text(_) => {panic!()}
        };
        self.unprocessed_tokens = UnprocessedTokens::Tokens(vec![token]);
        token
    }
}