#include "network.h"
#include <torch/torch.h>
#include <iostream>

using namespace torch;
using namespace std;

int main() {
	Net network(128,64);
	cout << network << "\n\n";
	Tensor x, out;
	x = torch::randn({2, 128});
	out = network->forward(x);
	cout << out;
}
