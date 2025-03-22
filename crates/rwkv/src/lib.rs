pub mod sampling;
pub mod tokenizer;
pub mod context_manager;

#[cfg(test)]
mod test_accuracy;

use burn::module::{Content, DisplaySettings, Module, ModuleDisplay, Param};
use burn::nn::{Linear, Sigmoid, Tanh, LinearConfig, Initializer};
use burn::prelude::{Tensor, Backend, Config, Device};
use burn::tensor::activation::{relu, sigmoid, softmax, softplus};
use burn::tensor::{Int, PrintOptions};
use burn::tensor::cast::ToElement;
use burn::tensor::module::embedding;


#[derive(Debug, Config)]
pub struct LayerNormConfig {
    /// The size of the input features.
    pub d_model: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
}

/// Applies Layer Normalization over an input tensor as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450).
///
/// `Y = norm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
/// Should be created using [LayerNormConfig](LayerNormConfig).
#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    /// The learnable weight.
    pub gamma: Param<Tensor<B, 1>>,
    /// The learnable bias.
    pub beta: Param<Tensor<B, 1>>,
    /// A value required for numerical stability.
    epsilon: f64,
}

impl LayerNormConfig {
    /// Initialize a new [layer norm](LayerNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        let gamma = Initializer::Ones.init([self.d_model], device);
        let beta = Initializer::Zeros.init([self.d_model], device);

        LayerNorm {
            gamma,
            beta,
            epsilon: self.epsilon,
        }
    }
}

impl<B: Backend> LayerNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See the [LayerNorm](LayerNorm) documentation for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any, d_model]`
    /// - output: `[..., any, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = input.clone().var_mean_bias(D - 1);
        
        //println!("var: {var}, mean: {mean}");
        let rstd_val = Tensor::ones(var.shape(), &input.device()).div(var.add_scalar(self.epsilon).sqrt());
        let scale = rstd_val.clone();
        let bias = -rstd_val*mean;
        //println!("scale: {scale}, bias: {bias}");
        
        let input_normalized = input.mul(scale).add(bias);

        //println!("gamma: {}, bias: {}", self.gamma.val(), self.beta.val());

        input_normalized
            .mul(self.gamma.val().unsqueeze())
            .add(self.beta.val().unsqueeze())
    }
}

#[derive(Debug, Config)]
pub struct GroupNormConfig {
    /// The number of groups to separate the channels into
    pub num_groups: usize,
    /// The number of channels expected in the input
    pub num_channels: usize,
    /// A value required for numerical stability. Default: 1e-5
    #[config(default = 1e-5)]
    pub epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    /// per-channel affine parameters initialized to ones (for weights)
    /// and zeros (for biases). Default: `true`
    #[config(default = true)]
    pub affine: bool,
}

/// Applies Group Normalization over a mini-batch of inputs as described in the paper [Group Normalization](https://arxiv.org/abs/1803.08494).
///
/// `Y = groupnorm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
/// Should be created using [GroupNormConfig](GroupNormConfig).
#[derive(Module, Debug)]
#[module(custom_display)]
pub struct GroupNorm<B: Backend> {
    /// The learnable weight
    pub gamma: Option<Param<Tensor<B, 1>>>,
    /// The learnable bias
    pub beta: Option<Param<Tensor<B, 1>>>,
    /// The number of groups to separate the channels into
    pub num_groups: usize,
    /// The number of channels expected in the input
    pub num_channels: usize,
    /// A value required for numerical stability
    pub epsilon: f64,
    /// A boolean value that when set to `true`, this module has learnable
    pub affine: bool,
}

impl<B: Backend> ModuleDisplay for GroupNorm<B> {
    fn custom_settings(&self) -> Option<DisplaySettings> {
        DisplaySettings::new()
            .with_new_line_after_attribute(false)
            .optional()
    }

    fn custom_content(&self, content: Content) -> Option<Content> {
        content
            .add("num_groups", &self.num_groups)
            .add("num_channels", &self.num_channels)
            .add("epsilon", &self.epsilon)
            .add("affine", &self.affine)
            .optional()
    }
}

impl GroupNormConfig {
    /// Initialize a new [group norm](GroupNorm) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> GroupNorm<B> {
        assert_eq!(
            self.num_channels % self.num_groups,
            0,
            "The number of channels must be divisible by the number of groups"
        );

        let (gamma, beta) = if self.affine {
            let gamma = Initializer::Ones.init([self.num_channels], device);
            let beta = Initializer::Zeros.init([self.num_channels], device);

            (Some(gamma), Some(beta))
        } else {
            (None, None)
        };

        GroupNorm {
            num_groups: self.num_groups,
            num_channels: self.num_channels,
            gamma,
            beta,
            epsilon: self.epsilon,
            affine: self.affine,
        }
    }
}

impl<B: Backend> GroupNorm<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// See [GroupNorm](GroupNorm) for more information.
    ///
    /// # Shapes
    ///
    /// - input: `[batch_size, num_channels, *]`
    /// - output: `[batch_size, num_channels, *]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        if input.shape().dims[1] != self.num_channels {
            panic!(
                "The number of channels in the input tensor should be equal to the number of channels in the GroupNorm module. Expected {}, got {}",
                self.num_channels,
                input.shape().dims[1]
            );
        }

        let gamma = self.gamma.as_ref().map(|x| x.val());
        let beta = self.beta.as_ref().map(|x| x.val());

        group_norm(
            input,
            gamma,
            beta,
            self.num_groups,
            self.epsilon,
            self.affine,
        )
    }
}

/// Applies Group Normalization over a mini-batch of inputs as described in the paper [Group Normalization](https://arxiv.org/abs/1803.08494).
///
/// `Y = groupnorm(X) * γ + β`
///
/// Where:
/// - `X` is the input tensor
/// - `Y` is the output tensor
/// - `γ` is the learnable weight
/// - `β` is the learnable bias
///
pub(crate) fn group_norm<B: Backend, const D: usize>(
    input: Tensor<B, D>,
    gamma: Option<Tensor<B, 1>>,
    beta: Option<Tensor<B, 1>>,
    num_groups: usize,
    epsilon: f64,
    affine: bool,
) -> Tensor<B, D> {
    if (beta.is_none() || gamma.is_none()) && affine {
        panic!("Affine is set to true, but gamma or beta is None");
    }

    let shape = input.shape();
    if shape.num_elements() <= 2 {
        panic!(
            "input rank for GroupNorm should be at least 3, but got {}",
            shape.num_elements()
        );
    }

    let batch_size = shape.dims[0];
    let num_channels = shape.dims[1];

    let hidden_size = shape.dims[2..].iter().product::<usize>() * num_channels / num_groups;
    let input = input.reshape([batch_size, num_groups, hidden_size]);

    let mean = input.clone().sum_dim(2) / hidden_size as f64;
    let input = input.sub(mean);

    let var = input.clone().powf_scalar(2.).sum_dim(2) / hidden_size as f64;
    let input_normalized = input.div(var.add_scalar(epsilon).sqrt());

    if affine {
        let mut affine_shape = [1; D];
        affine_shape[1] = num_channels;

        input_normalized
            .reshape(shape)
            .mul(gamma.clone().unwrap().reshape(affine_shape))
            .add(beta.clone().unwrap().reshape(affine_shape))
    } else {
        input_normalized.reshape(shape)
    }
}

#[derive(burn::prelude::Module, Debug)]
struct Embedding<B: Backend> {
    weight: Param<Tensor<B, 2>>
}

impl<B: Backend> Embedding<B> {
    pub fn new(config: &RWKV7Config, device: &B::Device) -> Self {
        let weight = Initializer::Zeros.init([config.vocab_size, config.d_model], device);
        Self { weight }
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        embedding(self.weight.val(), input)
    }
}

#[derive(burn::prelude::Module, Debug)]
struct TimeMixer<B: Backend> {
    x_r: Param<Tensor<B, 3>>,
    x_w: Param<Tensor<B, 3>>,
    x_k: Param<Tensor<B, 3>>,
    x_v: Param<Tensor<B, 3>>,
    x_a: Param<Tensor<B, 3>>,
    x_g: Param<Tensor<B, 3>>,
    w0: Param<Tensor<B, 3>>,
    r_k: Param<Tensor<B, 2>>,
    w1: Param<Tensor<B, 2>>,
    w2: Param<Tensor<B, 2>>,
    a1: Param<Tensor<B, 2>>,
    a2: Param<Tensor<B, 2>>,
    a0: Param<Tensor<B, 3>>,
    g1: Param<Tensor<B, 2>>,
    g2: Param<Tensor<B, 2>>,
    k_k: Param<Tensor<B, 3>>,
    k_a: Param<Tensor<B, 3>>,
    receptance: Linear<B>,
    key: Linear<B>,
    value: Linear<B>,
    output: Linear<B>,
    v2: Option<Param<Tensor<B, 2>>>,
    v1: Option<Param<Tensor<B, 2>>>,
    v0: Option<Param<Tensor<B, 3>>>,
    ln_x: GroupNorm<B>,
}

fn lerp<B: Backend, const D: usize>(start: Tensor<B, D>, end: Tensor<B, D>, weight: Tensor<B, D>) -> Tensor<B, D> {
    start.clone() + weight * ( end - start)
}

fn lora_forward<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x1 = x.matmul(l1.unsqueeze());
    let x = x1.matmul(l2.unsqueeze());
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn lora_forward_sigmoid<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.matmul(l1.unsqueeze());
    let activation = Sigmoid::new();
    let x = activation.forward(x).matmul(l2.unsqueeze());
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn lora_forward_tanh<B: Backend, const D: usize>(l1: Tensor<B, 2>, l2: Tensor<B, 2>, base: Option<Tensor<B, D>>, x: Tensor<B, D>) -> Tensor<B, D> {
    let x = x.matmul(l1.unsqueeze());
    let activation = Tanh::new();
    let x = activation.forward(x).matmul(l2.unsqueeze());
    if let Some(base) = base {
        x + base
    } else {
        x
    }
}

fn inner_norm<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize, p: f32) -> Tensor<B, D> {
    x.abs().powf_scalar(p).sum_dim(dim).powf_scalar(1./p)
}

fn normalize<B: Backend, const D: usize>(x: Tensor<B, D>, dim: usize, p: f32) -> Tensor<B, D> {
    // In python:
    /*
     eps = 1e-12
     denom = input.norm(p, dim, keepdim=True).clamp_min(eps).expand_as(input)
     return input / denom
     */
    let denom = inner_norm(x.clone(), dim, p).clamp_min(1e-12);
    x / denom
}

fn linear<B: Backend>(d_input: usize, d_output: usize, device: &B::Device) -> Linear<B> {
    LinearConfig::new(d_input, d_output).with_bias(false).init(device)
}

impl <B: Backend> TimeMixer<B> {


    fn new(layer_num: usize, config: &RWKV7Config, device: &B::Device) -> Self {

        struct LoraRanks {
            decay_lora: usize,
            iclr_lora: usize,
            v0_mix_amt_lora: usize,
            gate_lora: usize
        }
        let lora_ranks = if config.d_model < 2048 {
            LoraRanks {
                decay_lora: 64,
                iclr_lora: 64,
                v0_mix_amt_lora: 32,
                gate_lora: 128
            }
        } else if config.d_model < 4096 {
            LoraRanks {
                decay_lora: 128,
                iclr_lora: 64,
                v0_mix_amt_lora: 64,
                gate_lora: 256
            }
        } else if config.d_model < 6144 {
            LoraRanks {
                decay_lora: 192,
                iclr_lora: 96,
                v0_mix_amt_lora: 96,
                gate_lora: 384
            }
        } else {
            LoraRanks {
                decay_lora: 256,
                iclr_lora: 128,
                v0_mix_amt_lora: 128,
                gate_lora: 512
            }
        };
        let (v0, v1, v2) = if layer_num > 0 {
            (
                Some(Initializer::Zeros.init([1, 1, config.d_model], device)),
                Some(Initializer::Zeros.init([config.d_model, lora_ranks.v0_mix_amt_lora], device)),
                Some(Initializer::Zeros.init([lora_ranks.v0_mix_amt_lora, config.d_model], device))
            )
        } else {
            (None, None, None)
        };
        let ln_x = GroupNormConfig::new(config.n_heads, config.d_model).with_epsilon(1e-5*(config.d_model/config.n_heads) as f64).init(device);
        Self {
            x_r: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_w: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_k: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_v: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_a: Initializer::Zeros.init([1, 1, config.d_model], device),
            x_g: Initializer::Zeros.init([1, 1, config.d_model], device),
            w0: Initializer::Zeros.init([1, 1, config.d_model], device),
            r_k: Initializer::Zeros.init([config.n_heads, config.d_model/config.n_heads], device),
            w1: Initializer::Zeros.init([config.d_model, lora_ranks.decay_lora], device),
            w2: Initializer::Zeros.init([lora_ranks.decay_lora, config.d_model], device),
            a1: Initializer::Zeros.init([config.d_model, lora_ranks.iclr_lora], device),
            a2: Initializer::Zeros.init([lora_ranks.iclr_lora, config.d_model], device),
            a0: Initializer::Zeros.init([1, 1, config.d_model], device),
            g1: Initializer::Zeros.init([config.d_model, lora_ranks.gate_lora], device),
            g2: Initializer::Zeros.init([lora_ranks.gate_lora, config.d_model], device),
            k_k: Initializer::Zeros.init([1, 1, config.d_model], device),
            k_a: Initializer::Zeros.init([1, 1, config.d_model], device),
            receptance: linear(config.d_model, config.d_model, device),
            key: linear(config.d_model, config.d_model, device),
            value: linear(config.d_model, config.d_model, device),
            output: linear(config.d_model, config.d_model, device),
            v2,
            v1,
            v0,
            ln_x,
        }
    }

    fn forward(&self, hidden_state_in: Tensor<B, 3>, v0: Option<Tensor<B, 3>>, time_mixer_x_state: Tensor<B, 2>, vk_state: Tensor<B, 4>, d_model: usize, n_heads: usize) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 4>) {
        let d_batch = hidden_state_in.shape().dims[0];
        let d_tokens = hidden_state_in.shape().dims[1];
        let d_k = d_model/ n_heads;
        let d_v = d_k;
        assert_eq!(hidden_state_in.shape().dims[2], d_model);

        assert_eq!(time_mixer_x_state.shape().dims[0], d_batch);
        assert_eq!(time_mixer_x_state.shape().dims[1], d_model);

        assert_eq!(vk_state.shape().dims[0], d_batch);
        assert_eq!(vk_state.shape().dims[1], n_heads);
        assert_eq!(vk_state.shape().dims[2], d_v);
        assert_eq!(vk_state.shape().dims[3], d_k);

        let x_shifted_one_to_the_past : Tensor<B, 3> = if d_tokens > 1 {
            Tensor::cat(vec![time_mixer_x_state.unsqueeze(), hidden_state_in.clone().slice([None, Some((0, -1))])], 1)
        } else {
            time_mixer_x_state.unsqueeze()
        };

        let x_receptance = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_r.val());
        let x_decay = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_w.val());
        let x_key  = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_k.val());
        let x_value  = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_v.val());
        let x_iclr = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_a.val());
        let x_gate = lerp(hidden_state_in.clone(), x_shifted_one_to_the_past.clone(), self.x_g.val());

        let r = self.receptance.forward(x_receptance);
        let k = self.key.forward(x_key);
        let v = self.value.forward(x_value.clone());

        let (v0, v) = if let Some(v0) = v0 {
            let v0_mix = lora_forward(self.v1.clone().unwrap().val(), self.v2.clone().unwrap().val(), Some(self.v0.clone().unwrap().val()), x_value);
            (v0.clone(), lerp(v, v0, Sigmoid::new().forward(v0_mix)))
        } else {
            (v.clone(), v)
        };



        let gate = lora_forward_sigmoid(self.g1.val(), self.g2.val(), None, x_gate);

        let log_neglog_of_decay = lora_forward_tanh(self.w1.val(), self.w2.val(), Some(self.w0.val()), x_decay);
        let log_neglog_of_decay = (- softplus(-log_neglog_of_decay, 1.0)).add_scalar(-0.5);
        let log_of_decay = -log_neglog_of_decay.exp();
        let decay = log_of_decay.exp();

        let deformed_key = k.clone()*self.k_k.val();
        let deformed_key = normalize(deformed_key.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]), 3, 2.0);

        let iclr = sigmoid(lora_forward(self.a1.val(), self.a2.val(), Some(self.a0.val()), x_iclr));

        //println!("gate: {gate:?}");

        let k = lerp(k.clone(), k.clone()*iclr.clone(), self.k_a.val());

        // Separate into heads
        let r: Tensor<B, 4> = r.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let k: Tensor<B, 4> = k.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let v: Tensor<B, 4> = v.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let decay: Tensor<B, 4> = decay.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let iclr: Tensor<B, 4> = iclr.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);
        let deformed_key: Tensor<B, 4> = deformed_key.reshape([d_batch as i32, d_tokens as i32, n_heads as i32, -1]);

        //println!("r: {r:?}");
        //println!("w: {decay:?}");
        //println!("k: {k:?}");
        //println!("v: {v:?}");

        let mut vk_state = vk_state;
        let mut outputs = Vec::new();
        for t in 0..d_tokens {
            let ranges = [None, Some((t as i64, ((t)+1) as i64))];
            let r = r.clone().slice(ranges).squeeze(1);
            let k = k.clone().slice(ranges).squeeze(1);
            let v = v.clone().slice(ranges).squeeze(1);
            let decay = decay.clone().slice(ranges).squeeze(1);
            let iclr = iclr.clone().slice(ranges).squeeze(1);
            let deformed_key = deformed_key.clone().slice(ranges).squeeze(1);
            let (new_out, new_vk_state) = self.single_timestep(r, k, v, decay, iclr, deformed_key, vk_state);
            vk_state = new_vk_state;
            outputs.push(new_out);
        }
        let out: Tensor<B, 4> = Tensor::stack(outputs, 1);
        let out = out.reshape([d_batch, d_tokens, d_model]);
        //println!("a: {out}");

        //println!("ln_x gamma: {:?}", self.ln_x.gamma.clone().unwrap().val());
        //println!("ln_x beta: {:?}", self.ln_x.beta.clone().unwrap().val());

        // apply group normalization to each head and recombine the heads
        let out = self.ln_x.forward(out.reshape([(d_batch*d_tokens) as i32, d_model as i32])).reshape([d_batch as i32, d_tokens as i32, d_model as i32]);

        //println!("b: {out:}");

        let bonus = (r*k*self.r_k.val().unsqueeze()).sum_dim(3) * v;

        //println!("bonus: {bonus:?}");
        // Recombine bonus heads
        let bonus: Tensor<B, 3> = bonus.reshape([d_batch as i32, d_tokens as i32, d_model as i32]);
        let out = out + bonus;

        //println!("c: {out:?}");

        // Apply output gate
        let out = out*gate;

        //println!("d: {out:?}");

        // Project the output
        let out = self.output.forward(out);

        (out, v0, vk_state)
    }

    fn single_timestep(&self,
                       r: Tensor<B, 3>,
                       k: Tensor<B, 3>,
                       v: Tensor<B, 3>,
                       decay: Tensor<B, 3>,
                       iclr: Tensor<B, 3>,
                       deformed_key: Tensor<B, 3>,
                       vk_state: Tensor<B, 4>) -> (Tensor<B, 3>, Tensor<B, 4>) {

        let d_batch = vk_state.dims()[0];
        let d_heads = vk_state.dims()[1];
        let d_value = vk_state.dims()[2];
        let d_key = vk_state.dims()[3];

        assert_eq!(r.dims()[0], d_batch);
        assert_eq!(r.dims()[1], d_heads);
        assert_eq!(r.dims()[2], d_value);

        assert_eq!(k.dims()[0], d_batch);
        assert_eq!(k.dims()[1], d_heads);
        assert_eq!(k.dims()[2], d_key);

        assert_eq!(v.dims()[0], d_batch);
        assert_eq!(v.dims()[1], d_heads);
        assert_eq!(v.dims()[2], d_value);

        assert_eq!(decay.dims()[0], d_batch);
        assert_eq!(decay.dims()[1], d_heads);
        assert_eq!(decay.dims()[2], d_value);

        assert_eq!(iclr.dims()[0], d_batch);
        assert_eq!(iclr.dims()[1], d_heads);
        assert_eq!(iclr.dims()[2], d_value);

        assert_eq!(deformed_key.dims()[0], d_batch);
        assert_eq!(deformed_key.dims()[1], d_heads);
        assert_eq!(deformed_key.dims()[2], d_key);

        // transform inputs from BHK into column vectors BHK1, and put everything in float format for higher precision

        // TODO: to_full_precision() doesn't appear to exist, but should be used here
        let d = 3;
        let r = r.unsqueeze_dim(d);
        let k = k.unsqueeze_dim(d);
        let v = v.unsqueeze_dim(d);
        let decay = decay.unsqueeze_dim(d);
        let iclr = iclr.unsqueeze_dim(d);
        let deformed_key = deformed_key.unsqueeze_dim(d);

        // decay the kv state and remove the iclr amount of the value stored within the state at the deformed key
        let t_decay = decay.transpose();
        let vk_state = vk_state.clone() * t_decay - vk_state.matmul(deformed_key.clone()).matmul((iclr*deformed_key).transpose());

        // add in an dynamically iclr and 1-decay mixed amount of the latest value at the key
        // (key has been pre-adjusted in the calling code by the amount of iclr mixing)
        let vk_state = vk_state + (v.matmul(k.transpose()));

        // Apply receptance to the new stat
        let out = vk_state.clone().matmul(r);

        // Remove extra useless dimension from the output
        (out.squeeze(3), vk_state)
    }
}

#[derive(burn::prelude::Module, Debug)]
struct ChannelMixer<B: Backend> {
    x_k: Param<Tensor<B, 3>>,
    key: Linear<B>,
    value: Linear<B>,
}

impl<B: Backend> ChannelMixer<B> {
    fn new(config: &RWKV7Config, device: &B::Device) -> Self {
        Self {
            x_k: Initializer::Zeros.init([1, 1, config.d_model], device),
            key: linear(config.d_model, config.d_model*4, device),
            value: linear(config.d_model*4, config.d_model, device),
        }
    }

    fn forward(&self, hidden_state_in: Tensor<B, 3>, x_state: Tensor<B, 2>) -> Tensor<B, 3> {
        //let d_batch = hidden_state_in.shape().dims[0];
        let d_tokens = hidden_state_in.shape().dims[1];
        let x_shifted_one_to_the_past : Tensor<B, 3> = if d_tokens > 1 {
            Tensor::cat(vec![x_state.unsqueeze(), hidden_state_in.clone().slice([None, Some((0, -1))])], 1)
        } else {
            x_state.unsqueeze()
        };

        let x_in = lerp(hidden_state_in, x_shifted_one_to_the_past, self.x_k.val());
        let hidden = self.key.forward(x_in);
        let hidden = relu(hidden).powi_scalar(2);
        let out = self.value.forward(hidden);
        out
    }
}

#[derive(burn::prelude::Module, Debug)]
pub struct Block<B: Backend> {
    pub ln0: Option<LayerNorm<B>>,
    ln1: LayerNorm<B>,
    att: TimeMixer<B>,
    ln2: LayerNorm<B>,
    ffn: ChannelMixer<B>
}

impl<B: Backend> Block<B> {
    fn new( layer_num: usize, config: &RWKV7Config, device: &B::Device) -> Self {
        let ln_config = LayerNormConfig::new(config.d_model);
        let ln0 = if layer_num == 0 {
            Some(ln_config.init(device))
        } else {
            None
        };
        Block {
            ln0,
            ln1: ln_config.init(device),
            att: TimeMixer::new(layer_num, config, device),
            ln2: ln_config.init(device),
            ffn: ChannelMixer::new(&config, device)
        }
    }

    fn forward(&self, _layer_num: usize, x: Tensor<B, 3>, v0: Option<Tensor<B, 3>>, s: Option<&LayerState<B>>, d_model: usize, n_heads: usize) -> (Tensor<B, 3>, Tensor<B, 3>, LayerState<B>) {
        let num_tokens = x.dims()[1];

        use burn::tensor::{set_print_options, PrintOptions};

        let print_options = PrintOptions {
            precision: Some(4),
            threshold: 3000,
            ..Default::default()
        };

        set_print_options(print_options);
        
        let x = if let Some(ln0) = &self.ln0 {
            ln0.forward(x)
        } else {
            x
        };
        
        let s: LayerState<B> = if let Some(s) = s {
            s.clone()
        } else {
            LayerState::new_from_input(x.shape().dims[0], d_model, n_heads, &x.device())
        };
        let x1 = self.ln1.forward(x.clone());
        let new_time_mixer_x_state: Tensor<B, 2> = if num_tokens > 1 {
            x1.clone().slice([None, Some((-2, -1))])
        } else {
            x1.clone()
        }.squeeze(1);
        let (x2, new_v0, new_vk_state) = self.att.forward(x1.clone(), v0, s.time_mixer_x_state, s.vk_state, d_model, n_heads);
        let x3 = x + x2;
        let x4 = self.ln2.forward(x3.clone());
        let new_channel_mixer_x_state: Tensor<B, 2> = if num_tokens > 1 {
            x4.clone().slice([None, Some((-2, -1))])
        } else {
            x4.clone()
        }.squeeze(1);
        let x5 = self.ffn.forward(x4.clone(), s.channel_mixer_x_state);

        let x6 = x5 + x3;

        let new_s = LayerState {
            time_mixer_x_state: new_time_mixer_x_state,
            vk_state: new_vk_state,
            channel_mixer_x_state: new_channel_mixer_x_state
        };
        (x6, new_v0, new_s)
    }
}

#[derive(Clone, Debug)]
pub struct LayerState<B: Backend> {
    time_mixer_x_state: Tensor<B, 2>,
    vk_state: Tensor<B, 4>,
    channel_mixer_x_state: Tensor<B, 2>
}

impl<B: Backend> LayerState<B> {
    pub fn new_from_input(batch: usize, d_model: usize, n_heads: usize, device: &B::Device) -> Self {
        let k = d_model/n_heads;
        let v = k;
        let time_mixer_x_state = Tensor::zeros([batch, d_model], device);
        let vk_state = Tensor::zeros([batch, n_heads, v, k], device);
        let channel_mixer_x_state = Tensor::zeros([batch, d_model], device);
        LayerState {
            time_mixer_x_state,
            vk_state,
            channel_mixer_x_state
        }
    }
}

#[derive(Debug)]
pub struct RWKV7Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl RWKV7Config {
    pub fn rwkv_7_0b1 () -> Self {
        RWKV7Config {
            vocab_size: 65536,
            d_model: 768,
            n_layers: 12,
            n_heads: 12,
        }
    }

    pub fn rwkv_7_1b5 () -> Self {
        RWKV7Config {
            vocab_size: 65536,
            d_model: 2048,
            n_layers: 24,
            n_heads: 32,
        }
    }

    pub fn rwkv_7_2b9 () -> Self {
        RWKV7Config {
            vocab_size: 65536,
            d_model: 2560,
            n_layers: 32,
            n_heads: 40,
        }
    }

    pub fn from_record<B: Backend>(record: &RWKV7Record<B>) -> Self {
        let n_layers = record.blocks.len();
        let s: [usize; 2] = record.emb.weight.shape().dims();
        let vocab_size = s[0];
        let d_model = s[1];
        let s: [usize; 2] = record.blocks[0].att.r_k.shape().dims();
        let n_heads = s[0];
        RWKV7Config {
            vocab_size,
            d_model,
            n_layers,
            n_heads,
        }
    }
}

#[derive(burn::prelude::Module, Debug)]
pub struct RWKV7<B: Backend> {
    emb: Embedding<B>,
    pub blocks: Vec<Block<B>>,
    ln_out: LayerNorm<B>,
    head: Linear<B>,
    n_heads: usize,
    d_model: usize,
}

impl<B: Backend> RWKV7<B> {
    pub fn new(config: RWKV7Config, device: &B::Device) -> Self{
        let mut blocks = Vec::new();
        for i in 0..config.n_layers {
            blocks.push(Block::new(i, &config, device))
        }
        let ln_out_config = LayerNormConfig::new(config.d_model);
        let head_config = LinearConfig::new(config.d_model, config.vocab_size).with_bias(false);
        Self {
            emb: Embedding::new(&config, device),
            blocks,
            ln_out: ln_out_config.init(device),
            head: head_config.init(device),
            n_heads: config.n_heads,
            d_model: config.d_model
        }
    }

    pub fn do_greedy_sample(logits: Tensor<B, 3>) -> (Tensor<B, 1, Int>, u16) {
        let logits_transformed: Tensor<B, 1> = logits.squeeze::<2>(0).squeeze(0);
        let output_token = softmax(logits_transformed, 0).argmax(0);
        let output_u16 = output_token.to_data().as_slice::<B::IntElem>().unwrap()[0].to_u16();
        (output_token, output_u16)
    }

    pub fn forward(&self, input: Tensor<B, 2, Int>, channel_states: Option<&Vec<LayerState<B>>>) -> (Tensor<B, 3>, Vec<LayerState<B>>) {
        let mut x = self.emb.forward(input);

        //println!("after emb: {x:?}");
        
        let mut v0 = None;
        let mut new_channel_states = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let channel_state = if let Some(s) = channel_states{
                Some(&s[i])
            } else {
                None
            };
            let (new_x, new_v0, new_channel_state) = block.forward(i, x, v0, channel_state, self.d_model, self.n_heads);
            x = new_x;
            //println!("after block {i}: {x:?}");
            
            if i > 0 {
                //panic!()
            }
            v0 = Some(new_v0);
            new_channel_states.push(new_channel_state);
        }

        let x = self.ln_out.forward(x);
        //println!("after ln_out: {x:?}");
        
        
        let logits = self.head.forward(x);

        (logits, new_channel_states)
    }
    
    pub fn forward_from_vec(&self, inputs: &[u16], device: &Device<B>, channel_states: Option<&Vec<LayerState<B>>>) -> (Tensor<B, 3>, Vec<LayerState<B>>) {
        self.forward(Tensor::<B, 1, Int>::from_ints(&inputs[..], device).unsqueeze(), channel_states)
    }

    pub fn forward_from_vec_and_greedy_sample(&self, inputs: &[u16], device: &Device<B>, channel_states: Option<&Vec<LayerState<B>>>) -> (Tensor<B, 1, Int>, u16, Vec<LayerState<B>>) {
        let (logits, new_channel_states) = self.forward(Tensor::<B, 1, Int>::from_ints(&inputs[..], device).unsqueeze(), channel_states);
        let (new_token_tensor, new_token_value) = RWKV7::do_greedy_sample(logits.slice([0..1, (inputs.len()-1)..inputs.len()]));
        (new_token_tensor, new_token_value, new_channel_states)
    }
}
