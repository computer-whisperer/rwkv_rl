use cubecl::{cube, prelude::*, CubeType};

#[derive(CubeType, Clone, Hash, Eq, PartialEq, Debug)]
pub(crate) struct TimeMixKernelConfig {
    pub H: u32,
    pub C: u32,
    pub chunk_len: u32,
}


#[cube(launch)]
pub fn fused_time_mix_forward<F: Float>(
    state_in: &Tensor<f32>,
    state_out: &mut Tensor<f32>,
    r_in: &Tensor<F>,
    w_in: &Tensor<F>,
    k_in: &Tensor<F>,
    v_in: &Tensor<F>,
    a_in: &Tensor<F>,
    b_in: &Tensor<F>,
    sa_out: &mut Tensor<f32>,
    y_out: &mut Tensor<f32>,
    #[comptime] config: TimeMixKernelConfig
) {
    let T = w_in.shape(1);
    let hh = CUBE_POS_X;
    let bb = CUBE_POS_Y;
    let i = UNIT_POS_X;

    let mut state = Array::<f32>::new(config.C);
    let s_idx = bb*config.H*config.C*config.C + hh*config.C*config.C + i*config.C;
    #[unroll]
    for j in 0..config.C {
        state[j] = state_in[s_idx + j]
    }

    let mut r = SharedMemory::<f32>::new(config.C);
    let mut k = SharedMemory::<f32>::new(config.C);
    let mut w = SharedMemory::<f32>::new(config.C);
    let mut a = SharedMemory::<f32>::new(config.C);
    let mut b = SharedMemory::<f32>::new(config.C);

    for t in 0..T {
        let ind = bb*T*config.H*config.C + t*config.H*config.C + hh*config.C + i;

        sync_units();
        r[i] = f32::cast_from(r_in[ind]);
        k[i] = f32::cast_from(k_in[ind]);
        w[i] = Exp::exp(-Exp::exp(f32::cast_from(w_in[ind])));
        a[i] = f32::cast_from(a_in[ind]);
        b[i] = f32::cast_from(b_in[ind]);
        sync_units();

        let mut sa = 0.0f32;
        #[unroll]
        for j in 0..config.C {
            sa += a[j] * state[j];
        }
        sa_out[ind] = sa;

        let v = f32::cast_from(v_in[ind]);
        let mut y = 0.0f32;
        #[unroll]
        for j in 0..config.C {
            state[j] = state[j] * w[j] + sa * b[j] + k[j] * v;
            y += state[j] * r[j];
        }
        y_out[ind] = y;

/*
        if (t + 1)%config.chunk_len == 0 {
            let base = (bb*config.H + hh)*(T/config.chunk_len)*config.C*config.C + (t/config.chunk_len)*config.C*config.C + i;
            #[unroll]
            for j in 0..config.C {
                state_out[base + j*config.C] = state[j];
            }
        }*/
    }

    #[unroll]
    for j in 0..config.C {
        state_out[s_idx + j] = state[j];
    }
    sync_units();
}