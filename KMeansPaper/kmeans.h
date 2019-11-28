#pragma once
#include <thrust/device_vector.h>
#include "centroids.h"
#include "labels.h"
#include <thrust/reduce.h>

namespace kmeans {

template<typename T>
int kmeans(int iterations,
           int n, int d, int k,
           thrust::device_vector<T>& data,
           thrust::device_vector<int>& labels,
           thrust::device_vector<T>& centroids,
           thrust::device_vector<T>& distances,
           bool init_from_labels=true,
           double threshold=0.000001) {
    thrust::device_vector<T> data_dots(n);
    thrust::device_vector<T> centroid_dots(n);
    thrust::device_vector<T> pairwise_distances(n * k);
    
    detail::make_self_dots(n, d, data, data_dots);

    if (init_from_labels) {
        detail::find_centroids(n, d, k, data, labels, centroids);
    }   
    T prior_distance_sum = 0;
    int i = 0;
    for(; i < iterations; i++) {
        detail::calculate_distances(n, d, k,
                                    data, centroids, data_dots,
                                    centroid_dots, pairwise_distances);

        int changes = detail::relabel(n, k, pairwise_distances, labels, distances);
       
        
        detail::find_centroids(n, d, k, data, labels, centroids);
        T distance_sum = thrust::reduce(distances.begin(), distances.end());
        std::cout << "Iteration " << i << " produced " << changes
                  << " changes, and total distance is " << distance_sum << std::endl;

        if (i > 0) {
            T delta = distance_sum / prior_distance_sum;
            if (delta > 1 - threshold) {
                std::cout << "Threshold triggered, terminating iterations early" << std::endl;
                return i + 1;
            }
        }
        prior_distance_sum = distance_sum;
    }
    return i;
}

}
