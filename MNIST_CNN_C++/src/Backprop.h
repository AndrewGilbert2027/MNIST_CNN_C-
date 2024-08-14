#pragma once
#include "Conv2D.h"

namespace backprop
{
	/// Compute the derivative matrix of a single slice in one filter block w of W
	/// @param a 2-dimensional slice with dimensions (n_h_prev, n_w_prev)
	/// @param w 2-dimensional slice of dimension (f x f)
	/// @param dZ 2-dimensional slice of dimension (n_h, n_w)
	///	@param stride the stride length applied to convolution
	/// @return 2-dimensional array of w derivatives for backpropagation
	dim2 back_prop_slice_dW(const dim2& a, const dim2& w, const dim2& dZ, const size_t stride);

	/// Compute the derivative matrix of a single slice a
	///	@param w 2-dimensional filter slice of dimensions (f x f)
	///	@param a 2-dimensional filter slice of dimensions (n_h_prev, n_w_prev)
	///	@param dZ 2-dimensional slice of dimension (n_h, n_w)
	///	@param stride the stride length applied to convolution
	///	@return 2-dimensional array of a derivatives for backpropagation
	dim2 back_prop_slice_dA(const dim2& w, const dim2& a, const dim2& dZ, const size_t stride);

	/// Compute the 3-dimensional matrix of an input image with respect to one dZ of 2-dimensions
	/// @param a_i the ith training example input 
	/// @param w_c the c channel block in convolutional filter
	/// @param dZ 2-dimensional slice of the derivative matrix dZ
	/// @param stride the stride used for forward propagation
	/// @return 3-dimensional matrix dA of derivatives for the ith training input
	dim3 back_prop_slice_dA_3d(const dim3& a_i, const dim3& w_c, const dim2& dZ, const size_t stride);

	dim3 back_prop_slice_dW_3d(const dim3& a_i, const dim2& dZ_i_c, const size_t stride, const size_t f);

	/// Compute the derivative matrix for dA given the derivative of dZ
	///	@param W 4-dimensional array representing convolutional block of shape (f x f x n_c_prev x n_c)
	///	@param dZ 4-dimensional array representing a derivative matrix of shape (m x n_h x n_w x n_c)
	///	@param stride the stride applied to convolutional block
	///	@return 4-dimensional derivative matrix representing the derivative of an input block for each training iteration
	dim4 backprop_dA(const dim4& W, const dim4& dZ, const dim4& A, const size_t stride);

	/// 
	/// @param A The input array of size (m x n_h_prev x n_w_prev x n_c_prev)
	/// @param dZ 4-dimensional derivative matrix of shape (m x n_h x n_w x n_c)
	/// @param stride The current stride applied to the convolutional block
	///	@param f The horizontal and vertical length of a filter 
	/// @return 4-dimensional derivative matrix of a convolutional block
	dim4 backprop_dW(const dim4& A, const dim4& dZ, const size_t stride, const size_t f);

	dim4 backprop_dW(const dim4& A, const dim4& dZ, const size_t stride, const size_t f, const size_t padding);

	dim4 backprop_dA(const dim4& W, const dim4& dZ, const dim4& A, const size_t stride, const size_t padding);

	dim4 backprop_db(const dim4& dZ);

	dim2 get_2d_filter_slice(const dim3& w, const size_t n_c_prev);

	dim2 get_2d_input_slice(const dim3& a, const size_t n_c_prev);

	dim2 get_2d_Z_slice(const dim3& dZ_i, const size_t n_c);

	void move_derivative_values_at_depth(dim3& dA_i, dim2& dA_i_d, const size_t& d);

	void add_matrix_3d(dim3& anchor, const dim3& adder);

	void add_matrix_4d(dim4& anchor, const dim4& adder);

	dim2 subtract_matrix_2d(const dim2& anchor, const dim2& sub);

	void add_Wc_derivatives(dim4& dW, const dim3& dW_i_c, const size_t c);

	void divide_array(dim4& dW, float m);

	dim2 update_weights_dense(const dim2& W, const dim2& dW, const double& lr);
	dim1 update_biases_dense(const dim1& b, const dim1& db, const double& lr);

	dim4 update_weights(const dim4& W, const dim4& dW, const double& lr);
	dim4 update_biases(const dim4& b, const dim4& db, const double& lr);

}
