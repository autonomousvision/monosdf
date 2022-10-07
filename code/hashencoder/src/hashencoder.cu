#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include <torch/extension.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// requires CUDA >= 10 and ARCH >= 70
// this is very slow compared to float or __half2, do not use!
static inline  __device__ at::Half atomicAdd(at::Half *address, at::Half val) {
  return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
static inline __host__ __device__ T div_round_up(T val, T divisor) {
	return (val + divisor - 1) / divisor;
}


template <uint32_t D>
__device__ uint32_t fast_hash(const uint32_t pos_grid[D]) {
	static_assert(D <= 7, "fast_hash can only hash up to 7 dimensions.");

	// While 1 is technically not a good prime for hashing (or a prime at all), it helps memory coherence
	// and is sufficient for our use case of obtaining a uniformly colliding index from high-dimensional
	// coordinates.
    constexpr uint32_t primes[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

	uint32_t result = 0;
	#pragma unroll
	for (uint32_t i = 0; i < D; ++i) {
		result ^= pos_grid[i] * primes[i];
	}

	return result;
}


template <uint32_t D, uint32_t C>
__device__ uint32_t get_grid_index(const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[D]) {
	uint32_t stride = 1;
	uint32_t index = 0;

	#pragma unroll
    for (uint32_t d = 0; d < D && stride <= hashmap_size; d++) {
        index += pos_grid[d] * stride;
        // stride *= (resolution + 1);
        stride *= resolution;

    }

    if (stride > hashmap_size) {
        index = fast_hash<D>(pos_grid);
    }

	return (index % hashmap_size) * C + ch;
}

template <uint32_t D>
__device__ uint32_t get_grid_index(const uint32_t pos_grid[D]) {
	uint32_t stride = 1;
	uint32_t index = 0;

	#pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        index += pos_grid[d] * stride;
        stride *= 2;
    }
	return index;
}

__device__ inline float smoothstep(float val) {
	return val*val*(3.0f - 2.0f * val);
}

__device__ inline float smoothstep_derivative(float val) {
	return 6*val*(1.0f - val);
}

__device__ inline float identity_fun(float val) {
	return val;
}

__device__ inline float identity_derivative(float val) {
	return 1;
}

template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_grid(
    const scalar_t * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ outputs, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H,
    const bool calc_grad_inputs, 
    scalar_t * __restrict__ dy_dx
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b >= B) return;

    const uint32_t level = blockIdx.y;
    
    // locate
    grid += (uint32_t)offsets[level] * C;
    inputs += b * D;
    outputs += level * B * C + b * C;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            outputs[ch] = 0; 
        }
        if (calc_grad_inputs) {
            dy_dx += b * D * L * C + level * D * C; // B L D C
            #pragma unroll
            for (uint32_t d = 0; d < D; d++) {
                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    dy_dx[d * C + ch] = 0; 
                }       
            }
        }
        return;
    }

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;
    
    // calculate coordinate
    float pos[D];
    float pos_derivative[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = (float)inputs[d] * scale; // + 0.5f;
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        pos_derivative[d] = smoothstep_derivative(pos[d]);
        pos[d] = smoothstep(pos[d]);
    }

    // printf("[b=%d, l=%d] pos=(%f, %f, %f)+(%d, %d, %d), scale=%f\n", b, level, pos[0], pos[1], pos[2], pos_grid[0], pos_grid[1], pos_grid[2], scale);

    // interpolate
    scalar_t results[C] = {0}; // temp results in register

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);

        // writing to register (fast)
        #pragma unroll
        for (uint32_t ch = 0; ch < C; ch++) {
            results[ch] += w * grid[index + ch];
        }

        //printf("[b=%d, l=%d] int %d, idx %d, w %f, val %f, pos_grid=(%d %d %d)\n", 
        //b, level, idx, index, w, grid[index], pos_grid_local[0], pos_grid_local[1], pos_grid_local[]);
    }    

    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < C; ch++) {
        outputs[ch] = results[ch]; 
    }

    // prepare dy_dx for calc_grad_inputs
    // differentiable (soft) indexing: https://discuss.pytorch.org/t/differentiable-indexing/17647/9
    if (calc_grad_inputs) {

        dy_dx += b * D * L * C + level * D * C; // B L D C

        #pragma unroll
        for (uint32_t gd = 0; gd < D; gd++) {

            scalar_t results_grad[C] = {0};

            #pragma unroll
            for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
                float w = scale;
                uint32_t pos_grid_local[D];

                #pragma unroll
                for (uint32_t nd = 0; nd < D - 1; nd++) {
                    const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                    if ((idx & (1 << nd)) == 0) {
                        w *= 1 - pos[d];
                        pos_grid_local[d] = pos_grid[d];
                    } else {
                        w *= pos[d];
                        pos_grid_local[d] = pos_grid[d] + 1;
                    }
                }

                pos_grid_local[gd] = pos_grid[gd];
                uint32_t index_left = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);
                pos_grid_local[gd] = pos_grid[gd] + 1;
                uint32_t index_right = get_grid_index<D, C>(0, hashmap_size, resolution, pos_grid_local);

                #pragma unroll
                for (uint32_t ch = 0; ch < C; ch++) {
                    results_grad[ch] += w * (grid[index_right + ch] - grid[index_left + ch]) * pos_derivative[gd];
                }
            }

            #pragma unroll
            for (uint32_t ch = 0; ch < C; ch++) {
                dy_dx[gd * C + ch] = results_grad[ch];
            }
        }
    }
}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    scalar_t * __restrict__ grad_grid, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
	if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grid += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = (float)inputs[d] * scale;// + 0.5f;
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        pos[d] = smoothstep(pos[d]);
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
    }

    // interpolate
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        float w = 1;
        uint32_t pos_grid_local[D];

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = pos_grid[d] + 1;
            }
        }

        uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);

        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
            }
        }
    }    
}


template <typename scalar_t, uint32_t D, uint32_t C>
__global__ void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,  
    scalar_t * __restrict__ grad_inputs, 
    uint32_t B, uint32_t L
) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= B * D) return;

    const uint32_t b = t / D;
    const uint32_t d = t - b * D;

    dy_dx += b * L * D * C;

    scalar_t result = 0;
    
    # pragma unroll
    for (int l = 0; l < L; l++) {
        # pragma unroll
        for (int ch = 0; ch < C; ch++) {
            result += grad[l * B * C + b * C + ch] * dy_dx[l * D * C + d * C + ch];
        }
    }

    grad_inputs[t] = result;
}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_second_backward_grad(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    const scalar_t * __restrict__ grad_grad_inputs, 
    const scalar_t * __restrict__ dy_dx, 
    scalar_t * __restrict__ grad_grad, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
	if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

    // locate
    grad_grad += level * B * C + b * C + ch;;
    grad += level * B * C + b * C + ch; // L, B, C

    grad_grad_inputs += b*D;
    dy_dx += b * L*D*C + level * D * C + ch;

    
    scalar_t result[N_C] = {0};

    # pragma unroll
    for (int d = 0; d < D; d++) {
        # pragma unroll
        for (int c = 0; c < N_C; c++) {
            result[c] += grad_grad_inputs[d] * dy_dx[d*C+c];
        }
    }

    // write to global memory
    for (int c = 0; c < N_C; c++) {
        grad_grad[c] = result[c];
    }

    //printf("[b=%d, l=%d, ch=%d, D=%d, C=%d, N_C=%d, grad_grad=(%f, %f)]\n", b, level, ch, D, C, N_C, result[0], result[1]);
    
    // grad_grad_inputs: [B, D], float
// grad_grad: [L, B, C], float
// grad2_embedding: [s0, C], float
// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// dy_dx: [B, L * D * C]
// H: base resolution

    
}


template <typename scalar_t, uint32_t D, uint32_t C, uint32_t N_C>
__global__ void kernel_grid_second_backward_embedding(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ inputs, 
    const scalar_t * __restrict__ grid, 
    const int * __restrict__ offsets, 
    const scalar_t * __restrict__ grad_grad_inputs, 
    const scalar_t * __restrict__ dy_dx, 
    scalar_t * __restrict__ grad2_embeddings, 
    const uint32_t B, const uint32_t L, const float S, const uint32_t H
) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * N_C / C;
	if (b >= B) return;

    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * N_C - b * C;

// grad_grad_inputs: [B, D], float
// grad_grad: [L, B, C], float
// grad2_embedding: [s0, C], float
// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// dy_dx: [B, L * D * C]
// H: base resolution

    // locate
    grad2_embeddings += offsets[level] * C;
    inputs += b * D;
    grad += level * B * C + b * C + ch; // L, B, C
    grad_grad_inputs += b*D;

    const uint32_t hashmap_size = offsets[level + 1] - offsets[level];
    const float scale = exp2f(level * S) * H - 1.0f;
    const uint32_t resolution = (uint32_t)ceil(scale) + 1;

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[D];
    float pos_derivative[D];
    uint32_t pos_grid[D];

    #pragma unroll
    for (uint32_t d = 0; d < D; d++) {
        pos[d] = (float)inputs[d] * scale;// + 0.5f;
        pos_grid[d] = floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        pos_derivative[d] = smoothstep_derivative(pos[d]);
        pos[d] = smoothstep(pos[d]);
    }

    scalar_t grad_cur[N_C] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < N_C; c++) {
        grad_cur[c] = grad[c];
        
    }
    
    scalar_t grad2_input_cur[D] = {0}; // fetch to register
    for (uint32_t d = 0; d < D; d++){
        grad2_input_cur[d] = grad_grad_inputs[d];
    }

    //printf("[b=%d, l=%d, ch=%d, D=%d, C=%d, N_C=%d], pos(%f, %f, %f), grid(%d, %d, %d), grad_cur(%f, %f), grad2_input_cur(%f, %f, %f)\n", 
    //b, level, ch, D, C, N_C, pos[0], pos[1],pos[2], pos_grid[0], pos_grid[1], pos_grid[2], grad_cur[0], grad_cur[1], grad2_input_cur[0], grad2_input_cur[1], grad2_input_cur[2]);

    // grad2_embeddings cache
    // don't need N_C
    scalar_t grad_embeddings_cur[N_C *(1<<D)] = {0}; // cache [2^D * N_C]

    // compute gradients
    #pragma unroll
    for (uint32_t gd = 0; gd < D; gd++) {

        scalar_t results_grad[C] = {0};

        #pragma unroll
        for (uint32_t idx = 0; idx < (1 << (D - 1)); idx++) {
            // TODO do we need scale here?
            float w = scale;
            // float w = 1.;
            uint32_t pos_grid_local[D];

            #pragma unroll
            for (uint32_t nd = 0; nd < D - 1; nd++) {
                const uint32_t d = (nd >= gd) ? (nd + 1) : nd;

                if ((idx & (1 << nd)) == 0) {
                    w *= 1 - pos[d];
                    pos_grid_local[d] = 0;
                } else {
                    w *= pos[d];
                    pos_grid_local[d] = 1;
                }
            }

            pos_grid_local[gd] = 0;
            uint32_t index_left = get_grid_index<D>(pos_grid_local);
            pos_grid_local[gd] = 1;
            uint32_t index_right = get_grid_index<D>(pos_grid_local);

            //printf("[b=%d, l=%d, ch=%d, D=%d, C=%d, N_C=%d], grid(%d, %d, %d), index_left_right(%d, %d), scale=%f, w=%f\n", 
            //b, level, ch, D, C, N_C, pos_grid_local[0], pos_grid_local[1], pos_grid_local[2], index_left, index_right, scale, w);

            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                grad_embeddings_cur[index_right*N_C+c] += w * grad_cur[c] * grad2_input_cur[gd] * pos_derivative[gd];
                grad_embeddings_cur[index_left*N_C+c] -= w * grad_cur[c] * grad2_input_cur[gd] * pos_derivative[gd];
            }
        }
    }
    


    // write to global memory
    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << D); idx++) {
        uint32_t pos_grid_local[D];
        uint32_t cache_index = 0;
        uint32_t stride = 1;

        #pragma unroll
        for (uint32_t d = 0; d < D; d++) {
            if ((idx & (1 << d)) == 0) {
                pos_grid_local[d] = pos_grid[d];
            } else {
                pos_grid_local[d] = pos_grid[d] + 1;
                cache_index += stride;
            }
            stride *= 2;
        }
        //printf()
        uint32_t index = get_grid_index<D, C>(ch, hashmap_size, resolution, pos_grid_local);

        //printf("[b=%d, l=%d, ch=%d, D=%d, C=%d, N_C=%d], grid(%d, %d, %d), cache_index=%d, grad_embeddings_cur=(%f, %f)\n", 
        //    b, level, ch, D, C, N_C, pos_grid_local[0], pos_grid_local[1], pos_grid_local[2], cache_index, grad_embeddings_cur[cache_index*N_C], grad_embeddings_cur[cache_index * N_C+ 1]);


        // atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
        // TODO: use float which is better than __half, if N_C % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && N_C % 2 == 0) {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(1.0*grad_embeddings_cur[cache_index*N_C + c]), (__half)(1.0*grad_embeddings_cur[cache_index*N_C+ c + 1])};
                //__half2 v ={0, 0};
                atomicAdd((__half2*)&grad2_embeddings[index + c], v);
            }
        // float, or __half when N_C % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < N_C; c++) {
                atomicAdd(&grad2_embeddings[index + c], grad_embeddings_cur[cache_index*N_C + c]);
            }
        }
    }    
}


template <typename scalar_t, uint32_t D>
void kernel_grid_wrapper(const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx) {
    static constexpr uint32_t N_THREAD = 512;
	const dim3 blocks_hashgrid = { div_round_up(B, N_THREAD), L, 1 };
    switch (C) {
        case 1: kernel_grid<scalar_t, D, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        case 2: kernel_grid<scalar_t, D, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        case 4: kernel_grid<scalar_t, D, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        case 8: kernel_grid<scalar_t, D, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets, outputs, B, L, S, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}

// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// outputs: [L, B, C], float (L first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
// dy_dx: [B, L * D * C]
template <typename scalar_t>
void hash_encode_forward_cuda(const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx) {
    switch (D) {
        case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets, outputs, B, C, L, S, H, calc_grad_inputs, dy_dx); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets, outputs, B, C, L, S, H, calc_grad_inputs, dy_dx); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
    
}

template <typename scalar_t, uint32_t D>
void kernel_grid_backward_wrapper(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs) {
    static constexpr uint32_t N_THREAD = 256;
	const uint32_t N_C = std::min(2u, C); // n_features_per_thread
	const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        case 1: 
            kernel_grid_backward<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H); 
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 2: 
            kernel_grid_backward<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4: 
            kernel_grid_backward<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 8: 
            kernel_grid_backward<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_embeddings, B, L, S, H);
            if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// grad_embeddings: [sO, C]
// H: base resolution
template <typename scalar_t>
void hash_encode_backward_cuda(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, scalar_t *grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, scalar_t *dy_dx, scalar_t *grad_inputs) {
    switch (D) {
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, grad_embeddings, B, C, L, S, H, calc_grad_inputs, dy_dx, grad_inputs); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


template <typename scalar_t, uint32_t D>
void kernel_grid_second_backward_wrapper(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, const int *offsets, 
    const uint32_t B, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, 
    const scalar_t *dy_dx, const scalar_t *grad_grad_inputs, scalar_t *grad_grad, scalar_t *grad2_embeddings) {
    static constexpr uint32_t N_THREAD = 256;
	const uint32_t N_C = std::min(2u, C); // n_features_per_thread
	const dim3 blocks_hashgrid = { div_round_up(B * C / N_C, N_THREAD), L, 1 };
    switch (C) {
        /*
        case 1: 
            kernel_grid_second_backward_grad<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, S, H); 
            kernel_grid_second_backward_embedding<scalar_t, D, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, S, H); 
            // if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 1><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        */
        case 2: 
            kernel_grid_second_backward_grad<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, S, H);
            kernel_grid_second_backward_embedding<scalar_t, D, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, S, H);
            // if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 2><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        case 4: 
            kernel_grid_second_backward_grad<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, S, H);
            kernel_grid_second_backward_embedding<scalar_t, D, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, S, H);
            // if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 4><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        
        case 8: 
            kernel_grid_second_backward_grad<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad_grad, B, L, S, H);
            kernel_grid_second_backward_embedding<scalar_t, D, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets, grad_grad_inputs, dy_dx, grad2_embeddings, B, L, S, H);
            // if (calc_grad_inputs) kernel_input_backward<scalar_t, D, 8><<<div_round_up(B * D, N_THREAD), N_THREAD>>>(grad, dy_dx, grad_inputs, B, L);
            break;
        
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


// grad_grad_inputs: [B, D], float
// grad_grad: [L, B, C], float
// grad2_embedding: [s0, C], float
// grad: [L, B, C], float
// inputs: [B, D], float, in [0, 1]
// embeddings: [sO, C], float
// offsets: [L + 1], uint32_t
// dy_dx: [B, L * D * C]
// H: base resolution
template <typename scalar_t>
void hash_encode_second_backward_cuda(const scalar_t *grad, const scalar_t *inputs, const scalar_t *embeddings, 
    const int *offsets, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, 
    const bool calc_grad_inputs, const scalar_t *dy_dx, const scalar_t *grad_grad_inputs, scalar_t *grad_grad, scalar_t *grad2_embeddings) {
    switch (D) {
        case 2: kernel_grid_second_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets, B, C, L, S, H, calc_grad_inputs, dy_dx, grad_grad_inputs, grad_grad, grad2_embeddings); break;
        case 3: kernel_grid_second_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets, B, C, L, S, H, calc_grad_inputs, dy_dx, grad_grad_inputs, grad_grad, grad2_embeddings); break;
        default: throw std::runtime_error{"GridEncoding: C must be 1, 2, 4, or 8."};
    }
}


void hash_encode_forward(const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor outputs, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, at::Tensor dy_dx) {
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(outputs);
    CHECK_CUDA(dy_dx);
    
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(dy_dx);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(dy_dx);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    inputs.scalar_type(), "hash_encode_forward", ([&] {
        hash_encode_forward_cuda<scalar_t>(inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), outputs.data_ptr<scalar_t>(), B, D, C, L, S, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>());
    }));
}

void hash_encode_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, at::Tensor grad_embeddings, const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, const at::Tensor dy_dx, at::Tensor grad_inputs) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(grad_embeddings);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_inputs);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(grad_embeddings);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_inputs);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(grad_embeddings);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_inputs);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "hash_encode_backward", ([&] {
        hash_encode_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), offsets.data_ptr<int>(), grad_embeddings.data_ptr<scalar_t>(), B, D, C, L, S, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_inputs.data_ptr<scalar_t>());
    }));
    
}


void hash_encode_second_backward(const at::Tensor grad, const at::Tensor inputs, const at::Tensor embeddings, const at::Tensor offsets, 
    const uint32_t B, const uint32_t D, const uint32_t C, const uint32_t L, const float S, const uint32_t H, const bool calc_grad_inputs, 
    const at::Tensor dy_dx, const at::Tensor grad_grad_inputs, at::Tensor grad_grad, at::Tensor grad2_embeddings) {
    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets);
    CHECK_CUDA(dy_dx);
    CHECK_CUDA(grad_grad_inputs);
    CHECK_CUDA(grad_grad);
    CHECK_CUDA(grad2_embeddings);
    
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets);
    CHECK_CONTIGUOUS(dy_dx);
    CHECK_CONTIGUOUS(grad_grad_inputs);
    CHECK_CONTIGUOUS(grad_grad);
    CHECK_CONTIGUOUS(grad2_embeddings);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets);
    CHECK_IS_FLOATING(dy_dx);
    CHECK_IS_FLOATING(grad_grad_inputs);
    CHECK_IS_FLOATING(grad_grad);
    CHECK_IS_FLOATING(grad2_embeddings);


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "hash_encode_second_backward", ([&] {
        hash_encode_second_backward_cuda<scalar_t>(grad.data_ptr<scalar_t>(), inputs.data_ptr<scalar_t>(), embeddings.data_ptr<scalar_t>(), 
        offsets.data_ptr<int>(), B, D, C, L, S, H, calc_grad_inputs, dy_dx.data_ptr<scalar_t>(), grad_grad_inputs.data_ptr<scalar_t>(),
        grad_grad.data_ptr<scalar_t>(), grad2_embeddings.data_ptr<scalar_t>());
    }));
    
}
