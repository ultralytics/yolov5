/* reference: https://github.com/NVIDIA/retinanet-examples */

#include "inter_union_cuda.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
/*#include <cub/device/device_radix_sort.cuh>
#include <cub/iterator/counting_input_iterator.cuh>*/

using namespace std;

constexpr int   kTPB     = 64;  // threads per block
constexpr int   kCorners = 4;
constexpr int   kPoints  = 8;

class Vector {
public:
    __host__ __device__ Vector( );  // Default constructor
    __host__ __device__ ~Vector( );  // Deconstructor
    __host__ __device__ Vector( float2 const point );
    float2 const p;
    friend class Line;

private:
    __host__ __device__ float cross( Vector const v ) const;
};

Vector::Vector( ) : p( make_float2( 0.0f, 0.0f ) ) {}

Vector::~Vector( ) {}

Vector::Vector( float2 const point ) : p( point ) {}

float Vector::cross( Vector const v ) const {
    return ( p.x * v.p.y - p.y * v.p.x );
}

class Line {
public:
    __host__ __device__ Line( );  // Default constructor
    __host__ __device__ ~Line( );  // Deconstructor
    __host__ __device__ Line( Vector const v1, Vector const v2 );
    __host__ __device__ float call( Vector const v ) const;
    __host__ __device__ float2 intersection( Line const l ) const;

private:
    float const a;
    float const b;
    float const c;
};

Line::Line( ) : a( 0.0f ), b( 0.0f ), c( 0.0f ) {}

Line::~Line( ) {}

Line::Line( Vector const v1, Vector const v2 ) : a( v2.p.y - v1.p.y ), b( v1.p.x - v2.p.x ), c( v2.cross( v1 ) ) {}

float Line::call( Vector const v ) const {
    return ( a * v.p.x + b * v.p.y + c );
}

float2 Line::intersection( Line const l ) const {
    float w { a * l.b - b * l.a };
    return ( make_float2( ( b * l.c - c * l.b ) / w, ( c * l.a - a * l.c ) / w ) );
}

template<typename T>
__host__ __device__ void rotateLeft( T *array, int const &count ) {
    T temp = array[0];
    for ( int i = 0; i < count - 1; i++ )
        array[i] = array[i + 1];
    array[count - 1] = temp;
}

__host__ __device__ static __inline__ float2 padfloat2( float2 a, float2 b ) {
    float2 res;
    res.x = a.x + b.x;
    res.y = a.y + b.y;
    return res;
}

__device__ float IntersectionArea( float2 *mrect, float2 *mrect_shift, float2 *intersection ) {
    int count = kCorners;
    for ( int i = 0; i < kCorners; i++ ) {
        float2 intersection_shift[kPoints] {};
        for ( int k = 0; k < count; k++ )
            intersection_shift[k] = intersection[k];
        float line_values[kPoints] {};
        Vector const r1( mrect[i] );
        Vector const r2( mrect_shift[i] );
        Line const   line1( r1, r2 );
        for ( int j = 0; j < count; j++ ) {
            Vector const inter( intersection[j] );
            line_values[j] = line1.call( inter );
        }
        float line_values_shift[kPoints] {};

#pragma unroll
        for ( int k = 0; k < kPoints; k++ )
            line_values_shift[k] = line_values[k];
        rotateLeft( line_values_shift, count );
        rotateLeft( intersection_shift, count );
        float2 new_intersection[kPoints] {};
        int temp = count;
        count = 0;
        for ( int j = 0; j < temp; j++ ) {
            if ( line_values[j] <= 0 ) {
                new_intersection[count] = intersection[j];
                count++;
            }
            if ( ( line_values[j] * line_values_shift[j] ) <= 0 ) {
                Vector const r3( intersection[j] );
                Vector const r4( intersection_shift[j] );
                Line const Line( r3, r4 );
                new_intersection[count] = line1.intersection( Line );
                count++;
            }
        }
        for ( int k = 0; k < count; k++ )
            intersection[k] = new_intersection[k];
    }

    float2 intersection_shift[kPoints] {};

    for ( int k = 0; k < count; k++ )
        intersection_shift[k] = intersection[k];
    rotateLeft( intersection_shift, count );

    // Intersection
    float intersection_area = 0.0f;
    if ( count > 2 ) {
        for ( int k = 0; k < count; k++ )
            intersection_area +=
                intersection[k].x * intersection_shift[k].y - intersection[k].y * intersection_shift[k].x;
    }
    return ( abs( intersection_area / 2.0f ) );
}


/* Below functions are used for iou computation (polygon);
Boxes have shape nx8 and Anchors have mx8;
Return intersection and union of boxes[i, :] and anchors[j, :] with shape of (n, m).
*/
__global__ void inter_union_cuda_kernel(int const numBoxes, int const numAnchors,
  float2 const *b_box_vals, float2 const *a_box_vals, float *inters, float *unions) {
  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int combos = numBoxes * numAnchors;
  for ( int tid = t; tid < combos; tid += stride ) {
    float2 intersection[kPoints] { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
    float2 rect1[kPoints] {};
    float2 rect1_shift[kPoints] {};
    float2 rect2[kPoints] {};
    float2 rect2_shift[kPoints] {};
    float2 pad;
#pragma unroll
    for ( int b = 0; b < kCorners; b++ ) {
      if (b_box_vals[(static_cast<int>(tid/numAnchors) * kCorners + b)].x == a_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)].x)
        pad.x = 0.001f;
      else
        pad.x = 0.0f;
      if (b_box_vals[(static_cast<int>(tid/numAnchors) * kCorners + b)].y == a_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)].y)
        pad.y = 0.001f;
      else
        pad.y = 0.0f;
      intersection[b] = padfloat2( b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )], pad);
      rect1[b]        = b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )];
      rect1_shift[b]  = b_box_vals[( static_cast<int>( tid / numAnchors ) * kCorners + b )];
      rect2[b]        = a_box_vals[( tid * kCorners + b ) % ( numAnchors * kCorners )];
      rect2_shift[b]  = a_box_vals[( tid * kCorners + b ) % ( numAnchors * kCorners )];
    }
    rotateLeft( rect1_shift, 4 );
    rotateLeft( rect2_shift, 4 );
    float intersection_area = IntersectionArea( rect2, rect2_shift, intersection );
    // Union
    float rect1_area = 0.0f;
    float rect2_area = 0.0f;
#pragma unroll
    for ( int k = 0; k < kCorners; k++ ) {
        rect1_area += rect1[k].x * rect1_shift[k].y - rect1[k].y * rect1_shift[k].x;
        rect2_area += rect2[k].x * rect2_shift[k].y - rect2[k].y * rect2_shift[k].x;
    }
    float union_area = ( abs( rect1_area ) + abs( rect2_area ) ) / 2.0f;
    // float iou_val = intersection_area / ( union_area - intersection_area );
    // Write out answer
    inters[tid] = intersection_area;
    unions[tid] = union_area-intersection_area;
    /* if ( isnan( intersection_area ) && isnan( union_area ) ) {
        iou_vals[tid] = 1.0f;
    } else if ( isnan( intersection_area ) ) {
        iou_vals[tid] = 0.0f;
    } else {
        iou_vals[tid] = iou_val;
    }*/
  }
}


int inter_union( const void *const *inputs, void *const *outputs, int num_boxes, int num_anchors, cudaStream_t stream ) {
  auto boxes    = static_cast<const float2 *>( inputs[0] );
  auto anchors  = static_cast<const float2 *>( inputs[1] );
  auto inters = static_cast<float *>( outputs[0] );
  auto unions = static_cast<float *>( outputs[1] );
  int numSMs;
  cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 );
  int threadsPerBlock = kTPB;
  int blocksPerGrid   = numSMs * 10;
  inter_union_cuda_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>( num_anchors, num_boxes, anchors, boxes, inters, unions );
  return 0;
}


/* Below functions are used for loss computation (polygon);
For boxes and anchors having the same shape: nx8;
Return intersection and union of boxes[i, :] and anchors[i, :] with shape of (n, ).
*/
__global__ void b_inter_union_cuda_kernel(int const numAnchors,
  float2 const *b_box_vals, float2 const *a_box_vals, float *inters, float *unions) {
  int t      = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for ( int tid = t; tid < numAnchors; tid += stride ) {
    float2 intersection[kPoints] { -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
                                -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f };
    float2 rect1[kPoints] {};
    float2 rect1_shift[kPoints] {};
    float2 rect2[kPoints] {};
    float2 rect2_shift[kPoints] {};
    float2 pad;
#pragma unroll
    for ( int b = 0; b < kCorners; b++ ) {
      if (b_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)].x == a_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)].x)
        pad.x = 0.001f;
      else
        pad.x = 0.0f;
      if (b_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)].y == a_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)].y)
        pad.y = 0.001f;
      else
        pad.y = 0.0f;
      intersection[b] = padfloat2( b_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)], pad);
      rect1[b]        = b_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)];
      rect1_shift[b]  = b_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)];
      rect2[b]        = a_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)];
      rect2_shift[b]  = a_box_vals[(tid * kCorners + b) % (numAnchors * kCorners)];
    }
    rotateLeft( rect1_shift, 4 );
    rotateLeft( rect2_shift, 4 );
    float intersection_area = IntersectionArea( rect2, rect2_shift, intersection );
    // Union
    float rect1_area = 0.0f;
    float rect2_area = 0.0f;
#pragma unroll
    for ( int k = 0; k < kCorners; k++ ) {
        rect1_area += rect1[k].x * rect1_shift[k].y - rect1[k].y * rect1_shift[k].x;
        rect2_area += rect2[k].x * rect2_shift[k].y - rect2[k].y * rect2_shift[k].x;
    }
    float union_area = ( abs( rect1_area ) + abs( rect2_area ) ) / 2.0f;
    // Write out answer
    inters[tid] = intersection_area;
    unions[tid] = union_area-intersection_area;
  }
}


int b_inter_union( const void *const *inputs, void *const *outputs, int num_anchors, cudaStream_t stream ) {
  auto boxes    = static_cast<const float2 *>( inputs[0] );
  auto anchors  = static_cast<const float2 *>( inputs[1] );
  auto inters = static_cast<float *>( outputs[0] );
  auto unions = static_cast<float *>( outputs[1] );
  int numSMs;
  cudaDeviceGetAttribute( &numSMs, cudaDevAttrMultiProcessorCount, 0 );
  int threadsPerBlock = kTPB;
  int blocksPerGrid   = numSMs * 10;
  b_inter_union_cuda_kernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>( num_anchors, anchors, boxes, inters, unions );
  return 0;
}
