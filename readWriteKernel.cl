__kernel void readWriteKernel(__global int* input_output) {
	uint idx = get_global_id(0);
	input_output[idx] = input_output[idx] + 5;
}
