#include <thrust/device_vector.h>
#include "kmeans.h"
#include "timer.h"
#include <iostream>
#include <cstdlib>
#include <typeinfo>

template<typename T>
void print_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            typename T::value_type value = array[i * n + j];
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
}

template<typename T>
void fill_array(T& array, int m, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            array[i * n + j] = (i % 2)*3 + j;
        }
    }
}

template<typename T>
void random_data(thrust::device_vector<T>& array, int m, int n) {
    thrust::host_vector<T> host_array(m*n);
    for(int i = 0; i < m * n; i++) {
        host_array[i] = (T)rand()/(T)RAND_MAX;
    }
    array = host_array;
}

void random_labels(thrust::device_vector<int>& labels, int n, int k) {
    thrust::host_vector<int> host_labels(n);
    for(int i = 0; i < n; i++) {
        host_labels[i] = rand() % k;
    }
    labels = host_labels;
}


void tiny_test() {
    int iterations = 1;
    int n = 5;
    int d = 3;
    int k = 2;

    
    thrust::device_vector<double> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(k * d);
    thrust::device_vector<double> distances(n);
    
    fill_array(data, n, d);
    std::cout << "Data: " << std::endl;
    print_array(data, n, d);

    labels[0] = 0;
    labels[1] = 0;
    labels[2] = 0;
    labels[3] = 1;
    labels[4] = 1;

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);
    
    int i = kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances);
    std::cout << "Desempenho " << i << " iterations" << std::endl;

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);

    std::cout << "Centroides:" << std::endl;
    print_array(centroids, k, d);

    std::cout << "Distancias:" << std::endl;
    print_array(distances, n, 1);

}


void more_tiny_test() {
	double dataset[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5,
		1.1, 1.2,
		0.5, 15.5,
		1.5, 15.5,
		1.5, 16.5,
		0.5, 16.5,
		1.2, 16.1,
		15.5, 15.5,
		16.5, 15.5,
		16.5, 16.5,
		15.5, 16.5,
		15.6, 16.2,
		15.5, 0.5,
		16.5, 0.5,
		16.5, 1.5,
		15.5, 1.5,
		15.7, 1.6};
	double centers[] = {
		0.5, 0.5,
		1.5, 0.5,
		1.5, 1.5,
		0.5, 1.5};
	 
    int iterations = 3;
    int n = 20;
    int d = 2;
    int k = 4;
	
	thrust::device_vector<double> data(dataset, dataset+n*d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<double> centroids(centers, centers+k*d);
    thrust::device_vector<double> distances(n);
    
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances, false);

    std::cout << "Labels: " << std::endl;
    print_array(labels, n, 1);

    std::cout << "Centroids:" << std::endl;
    print_array(centroids, k, d);

}

template<typename T>
void huge_test() {

    int iterations = 50;
    int n = 1e6;
    int d = 64;
    int k = 128;

    thrust::device_vector<T> data(n * d);
    thrust::device_vector<int> labels(n);
    thrust::device_vector<T> centroids(k * d);
    thrust::device_vector<T> distances(n);
    
    std::cout << "Genarando datos randoms" << std::endl;
    std::cout << "Numero de puntos: " << n << std::endl;
    std::cout << "Numero de dimensiones " << d << std::endl;
    std::cout << "Numero de clusters: " << k << std::endl;
    std::cout << "Numero de iteraciones: " << iterations << std::endl;
    std::cout << "Precision: " << typeid(T).name() << std::endl;
    
    random_data(data, n, d);
    random_labels(labels, n, k);
    kmeans::timer t;
    t.start();
    kmeans::kmeans(iterations, n, d, k, data, labels, centroids, distances);
    float time = t.stop();
    std::cout << "  Time: " << time/1000.0 << " s" << std::endl;

}

int main() {
    std::cout << "Eliga que tipo de test quiere:" << std::endl;
    std::cout << "Pequenho test: t" << std::endl;
    std::cout << "Otro pequenho test: m" << std::endl;
    std::cout << "Test grande: h: " << std::endl;
    char c;
    std::cin >> c;
    switch (c) {
    case 't':
        tiny_test();
        exit(0);
    case 'm':
        more_tiny_test();
        exit(0);
    case 'h':
        break;
    default:
        std::cout << "Corriendo test grande" << std::endl;
    }
    std::cout << "Doble precision (d) o simple precision (f): " << std::endl;
    std::cin >> c;
    switch(c) {
    case 'd':
        huge_test<double>();
        exit(0);
    case 'f':
        break;
    default:
        std::cout << "Corriendo presicion simple"
                  << std::endl;
    }
    huge_test<float>();
    
}
