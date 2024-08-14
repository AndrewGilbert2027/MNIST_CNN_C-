#pragma once
#include "Conv2D.h"


namespace pooling
{
	/// Given an input array, compute output array after applying a max pooling layer of size f and stride s
	/// @param A_prev Current input array of size (m, n_h_prev, n_w_prev, n_c)
	/// @param f Size of pooling filter (f x f)
	/// @param stride Number of steps to take after getting max in current slice
	/// @return 4-dimensional array of size (m, n_h, n_w, n_c) where m is number of training examples
	dim4 pool_forward_max(const dim4& A_prev, const size_t& f, const size_t& stride);

	/// Return 2-dimensional slice of 3d block with bounding coordinates and channel depth
	/// @param arr 3-dimensional array of size (n_h, n_w, n_c)
	/// @param v_start Starting 0 based index for vertical axis=0
	/// @param v_end  Ending index for vertical axis=0
	/// @param h_start Starting index for horizontal axis=1
	/// @param h_end Ending index for horizontal axis=1
	/// @param c Current channel to look at axis=2
	/// @return Current 2d slice of 3d cube
	dim2 get_slice_pooling(const dim3& arr, const size_t& v_start, const size_t& v_end, const size_t& h_start, const size_t& h_end, const size_t& c);

	/// Returns the maximum floating point number in 2-dimensional array
	/// @param arr 2 dimensional array of size (v_end - v_start, h_end - h_start)
	/// @return The maximum element in 2-dimensional  array
	double get_max_in_slice(const dim2& arr);

	///===========================///
	/// Backpropagation functions ///
	///===========================///

	struct coords
	{
		size_t x;
		size_t y;
		coords(): x(0), y(0) {}
		coords(const size_t x_s, const size_t y_s): x(x_s), y(y_s) {}
		coords(const coords& other)
		{
			this->x = other.x;
			this->y = other.y;
		}
		coords& operator=(const coords& other)
		{
			// Avoid copy assignment
			if (this != &other)
			{
				this->x = other.x;
				this->y = other.y;
			}
			return *this;
		}
	};

	coords get_max_coords(const dim2& array);
	dim4 compute_dA_pool_max(const dim4& A, const dim4& dA_pool, const size_t stride, const size_t f);
}