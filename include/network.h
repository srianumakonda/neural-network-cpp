#pragma once
#include <iostream>
#include <torch/torch.h>

struct NetImpl : torch::nn::Module {
	NetImpl(int fc1_dim, int fc2_dim)
		: fc1(fc1_dim, fc1_dim), fc2(fc1_dim, fc2_dim), out(fc2_dim, 1) {
			register_module("fc1", fc1);
			register_module("fc2", fc2);
			register_module("out", out);
	 	} 

	torch::Tensor forward(torch::Tensor x){
		x = torch::relu(fc1(x));
		x = torch::relu(fc2(x));
		x = out(x);
		return x;
	}

	torch::nn::Linear fc1{nullptr}, fc2{nullptr}, out{nullptr};
};

TORCH_MODULE(Net);

