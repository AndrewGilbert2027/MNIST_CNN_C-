#pragma once
#include <vector>

using dim4 = std::vector<std::vector<std::vector<std::vector<double>>>>;
using dim3 = std::vector<std::vector<std::vector<double>>>;
using dim2 = std::vector<std::vector<double>>;
using dim1 = std::vector<double>;

namespace convolutional
{
	/// @param X, input vector of shape (m, n_H, n_W, n_C) 
	/// @param pad, the amount of zero padding to add to the edges of the vector
	///	@return X_pad, output vector with padding of shape (m, n_H + pad, n_W + pad, n_C)
	dim4 zero_pad(const dim4& X, const size_t& pad);

	/// Given an input array and coordinates specifying the slice to take, return the sliced array
	/// @param arr 
	/// @param v_start 
	/// @param v_end 
	/// @param h_start 
	/// @param h_end 
	/// @return 
	dim3 get_slice(const dim3& arr, const size_t& v_start, const size_t& v_end, const size_t& h_start, const size_t& h_end);

	/// 
	/// @param filter -- convolutional filter block with shape (f, f, n_c_prev, n-c)
	/// @param c -- the current channel of filter to receive;
	/// @return filter_slice -- the current block of filters to apply to image
	dim3 get_filter_slice(const dim4& filter, const size_t& c);

	/// 
	/// @param a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
	/// @param W -- Single filter block of shape (f, f, n_C_prev)
	/// @param b -- bias parameter
	/// @return Z -- a scalar value of type, the result of convolving the sliding window (W, b) on a slice x of the input data
	double conv_single_step(const dim3& a_slice_prev, const dim3& W, double b);

	/// 
	/// @param A_prev -- the previous convolutional block output with shape (m, n_h_prev, n_w_prev, n_c_prev)
	/// @param W -- The filters of the current convolutional block with shape (f, f, n_c_prev, n_c)
	/// @param b -- The bias of the current convolutional block with shape (1, 1, 1, n_c)
	/// @param stride -- The stride of the current convolutional block
	/// @param pad -- The padding of the current convolutional block
	/// @return Z -- convolutional output with shape (m, n_h, n_w, n_c)
	dim4 conv_forward(const dim4& A_prev, const dim4& W, const dim4& b, const size_t& stride, const size_t& pad);

	/// 
	/// @param A arr of dimension (m x n_h x n_w x n_c)
	/// @return 2-dimensional array of size (m x (n_h x n_w x n_c))
	dim2 flatten(const dim4& A);

	dim4 reshape_from_flatten(const dim2& dA5, const dim4& A4);

	/// Applies the Rectified linear unit (ReLu) activation function towards a matrix.
	/// @param A 4-dimensional matrix of shape (m x n_h x n_w x n_c).
	/// @return 4-dimensional matrix of same shape after applying Relu activation function.
	dim4 relu(const dim4& A);

	dim4 relu_derivative(const dim4& input);
	dim4 relu_derivative(const dim4& A, const dim4& Z);
	dim3 remove_padding(const dim3& input, const size_t padding);
	dim4 pad(const dim4& input, const size_t padding);
}