//! RWKV v6 model implementation.
//!
//! The [RWKV model](https://wiki.rwkv.com/) is a recurrent neural network model
//! with performance on par with transformer architectures. Several variants are
//! available, candle implements the v5 and v6 versions and can be used with
//! Eagle 7B([blog post](https://blog.rwkv.com/p/eagle-7b-soaring-past-transformers)).
//!
//! Key characteristics:
//! - Linear attention mechanism
//! - Time-mixing for temporal dependencies
//! - Group normalization
//! - Feed forward gating
//! - State recycling for efficient inference
//!
//! # Example
//!
//! ```bash
//! cargo run --example rwkv --release -- \
//!   --prompt "The smallest prime is "
//!
//! > avx: true, neon: false, simd128: false, f16c: true
//! > temp: 0.00 repeat-penalty: 1.10 repeat-last-n: 64
//! > The smallest prime is ϕ(2) = 2.
//! > The smallest composite is ϕ(3) = 3.
//! > The smallest perfect number is ϕ(5) = 5.
//! > The smallest perfect square is ϕ(4) = 4.
//! > The smallest perfect cube is ϕ(6) = 6.
//! ```

use candle_transformers::models::with_tracing::{layer_norm, linear_no_bias, LayerNorm, Linear};
use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{embedding, Embedding, LayerNormConfig, Module, VarBuilder};


pub struct Config {
    hidden_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_heads: usize
}

impl Config {
    pub fn new_1B5() -> Self {
        Self {
            hidden_size: 2048,
            vocab_size: 2048,
            num_hidden_layers: 24,
            num_heads:
        }
    }
}

pub struct TimeMixer {
    prenorm: LayerNorm,
    tokenshift_r: Tensor,
    tokenshift_w: Tensor,
    tokenshift_k: Tensor,
    tokenshift_v: Tensor,
    tokenshift_a: Tensor,
    tokenshift_g: Tensor,
    receptance: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    bonus: Tensor,
    group_norm: LayerNorm
}

impl TimeMixer {
    fn new(block_index: usize, config: &Config, vb: VarBuilder) -> Result<Self> {
        let local_vb = vb.pp("att");

        Ok(Self {
            prenorm: layer_norm(config.hidden_size, 1e-5, vb.pp("ln1"))?,
            tokenshift_r: local_vb.get((1, 1, config.hidden_size), "time_maa_r")?,
            tokenshift_w: local_vb.get((1, 1, config.hidden_size), "time_maa_w")?,
            tokenshift_k: local_vb.get((1, 1, config.hidden_size), "time_maa_k")?,
            tokenshift_v: local_vb.get((1, 1, config.hidden_size), "time_maa_v")?,
            tokenshift_a: local_vb.get((1, 1, config.hidden_size), "time_maa_a")?,
            tokenshift_g: local_vb.get((1, 1, config.hidden_size), "time_maa_g")?,
            receptance: linear_no_bias(config.hidden_size, config.hidden_size, local_vb.pp("receptance"))?,
            key: linear_no_bias(config.hidden_size, config.hidden_size, local_vb.pp("key"))?,
            value: linear_no_bias(config.hidden_size, config.hidden_size, local_vb.pp("value"))?,
            output: linear_no_bias(config.hidden_size, config.hidden_size, local_vb.pp("output"))?,
            bonus: local_vb.get((1, 1, config.num_heads, config.hidden_size/config.num_heads), "time_faaaa")?,
            group_norm: layer_norm(config.hidden_size, 1e-5, vb.pp("ln3"))?
        })
    }
}

pub struct ChannelMixer {
    prenorm: LayerNorm,
    w_in: Linear,
    w_out: Linear,
    tokenshift: Tensor
}

impl ChannelMixer {
    fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let local_vb = vb.pp("ffn");
        let prenorm = layer_norm(config.hidden_size, 1e-5, vb.pp("ln2"))?;
        let w_in = linear_no_bias(config.hidden_size, config.hidden_size*4, local_vb.pp("key"))?;
        let w_out = linear_no_bias(config.hidden_size*4, config.hidden_size, local_vb.pp("value"))?;
        let tokenshift = local_vb.get((1, 1, config.hidden_size), "time_maa_k")?;
        Ok(Self {
            prenorm,
            w_in,
            w_out,
            tokenshift
        })
    }
}

#[derive(Debug, Clone)]
pub struct Block {
    time_mixer: TimeMixer,
    channel_mixer: ChannelMixer
}

impl Block {
    pub fn new(block_index: usize, config: &Config, vb: VarBuilder) -> Result<Self> {
        let time_mixer = TimeMixer::new(block_index, config, vb.clone());
        let channel_mixer = ChannelMixer::new(config, vb)?;
        Ok(Self { time_mixer, channel_mixer })
    }
}


#[derive(Debug, Clone)]
pub struct Model {
    embeddings: Embedding,
    embed_norm: LayerNorm,
    blocks: Vec<Block>,
    lm_head_norm: LayerNorm,
    lm_head_unembed: Linear,
}

impl Model {
    pub fn new(config: &Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("rwkv");
        let vb_b = vb_m.pp("blocks");
        let embeddings = embedding(config.vocab_size, config.hidden_size, vb_m.pp("embeddings"))?;
        let embed_norm = layer_norm(config.hidden_size, 1e-5, vb_b.pp(0).pp("ln0"))?;
        let mut blocks = Vec::with_capacity(config.num_hidden_layers);
        for block_index in 0..config.num_hidden_layers {
            let block = Block::new(block_index, config, vb_b.pp(block_index))?;
            blocks.push(block)
        }
        let lm_head_norm = layer_norm(config.hidden_size, 1e-5, vb_m.pp("ln_out"))?;
        let lm_head_unembed = linear_no_bias(config.hidden_size, config.vocab_size, vb.pp("head"))?;
        Ok(Self {
            embeddings,
            embed_norm,
            blocks,
            lm_head_norm,
            lm_head_unembed,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        let (_b_size, _seq_len) = xs.dims2()?;
        let mut xs = xs.apply(&self.embeddings)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            xs = block.forward(&xs, state)?;
            if self.layers_are_rescaled && (block_idx + 1) % self.rescale_every == 0 {
                xs = (xs / 2.)?
            }
        }
        let xs = xs.apply(&self.ln_out)?.apply(&self.head)?;
        state.pos += 1;
        Ok(xs)
    }
}
