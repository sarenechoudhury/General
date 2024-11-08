#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "hw3.h"  

Test(all_ofty, all_ofty_test_case_1) {
    unsigned int arr[] = {143, 134, 431, 314};
    unsigned int result = all_ofty(arr, 4);
    cr_assert(result == 1, "Expected true, got %d", result);
}

Test(all_ofty, all_ofty_test_case_2) {
    unsigned int arr[] = {123, 134, 431, 314};
    unsigned int result = all_ofty(arr, 4);
    cr_assert(result == 0, "Expected true, got %d", result);
}

Test(exists_ofty, exists_ofty_test_case_1) {
    unsigned int arr[] = {143, 134, 431, 314};
    unsigned int result = exists_ofty(arr, 4);
    cr_assert(result == 1, "Expected true, got %d", result);
}

Test(exists_ofty, exists_ofty_test_case_2) {
    unsigned int arr[] = {123, 134, 431, 314};
    unsigned int result = exists_ofty(arr, 4);
    cr_assert(result == 1, "Expected true, got %d", result);
}

Test(first_ofty, first_ofty_test_case_1) {
    unsigned int arr[] = {123, 134, 431, 314};
    unsigned int result = first_ofty(arr, 4);
    cr_assert(result == 1, "Expected 1, got %d", result);
}

Test(first_ofty, first_ofty_test_case_2) {
    unsigned int arr[] = {123, 124, 431, 314};
    unsigned int result = first_ofty(arr, 4);
    cr_assert(result == 2, "Expected 2, got %d", result);
}

Test(number_ofty, number_ofty_test_case_1) {
    unsigned int arr[] = {143, 134, 431, 314};
    unsigned int result = number_ofty(arr, 4);
    cr_assert(result == 4, "Expected 4, got %d", result);
}

Test(number_ofty, number_ofty_test_case_2) {
    unsigned int arr[] = {123, 134, 431, 314};
    unsigned int result = number_ofty(arr, 4);
    cr_assert(result == 3, "Expected 3, got %d", result);
}

Test(progress, progress_test_case_1) {
    unsigned int arr[] = {143, 134, 141, 314};
    unsigned int result[] = {144, 0, 142, 0};
    progress(arr, 4);
    for (int i = 0; i < 4; i++){
        cr_assert(arr[i] == result[i], "Expected %u, but got %u at index %d", result[i], arr[i], i);
    }
}

Test(progress, progress_test_case_2) {
    unsigned int arr[] = {143, 142, 141, 314};
    unsigned int result[] = {144, 143, 142, 0};
    progress(arr, 4);
    for (int i = 0; i < 4; i++){
        cr_assert(arr[i] == result[i], "Expected %u, but got %u at index %d", result[i], arr[i], i);
    }
}

Test(running_ofty, running_ofty_test_case_1) {
    unsigned int arr[] = {143, 134, 141, 314};
    unsigned int expected[] = {1, 2, 2, 3}; 
    unsigned int length = sizeof(arr) / sizeof(arr[0]);
    unsigned int* result = running_ofty(arr, length);
    for (unsigned int i = 0; i < length; i++) {
        cr_assert(result[i] == expected[i], "At index %u, expected %u but got %u", i, expected[i], result[i]);
    }
    free(result);
}

Test(running_ofty, running_ofty_test_case_2) {
    unsigned int arr[] = {123, 134, 141, 314};
    unsigned int expected[] = {0, 1, 1, 2}; 
    unsigned int length = sizeof(arr) / sizeof(arr[0]);
    unsigned int* result = running_ofty(arr, length);
    for (unsigned int i = 0; i < length; i++) {
        cr_assert(result[i] == expected[i], "At index %u, expected %u but got %u", i, expected[i], result[i]);
    }
    free(result);
}

Test(rotate_right, rotate_right_test_case_1) {
    int arr[] = {1, 2, 3, 4};
    unsigned int expected[] = {4, 1, 2, 3}; 
    unsigned int length = sizeof(arr) / sizeof(arr[0]);
    rotate_right(arr, length);
    for (unsigned int i = 0; i < length; i++) {
        cr_assert(arr[i] == expected[i], "At index %u, expected %u but got %u", i, expected[i], arr[i]);
    }
}

Test(rotate_right, rotate_right_test_case_2) {
    int arr[] = {4, 5, 60, 2435245, 19, 2};
    unsigned int expected[] = {2, 4, 5, 60, 2435245, 19}; 
    unsigned int length = sizeof(arr) / sizeof(arr[0]);
    rotate_right(arr, length);
    for (unsigned int i = 0; i < length; i++) {
        cr_assert(arr[i] == expected[i], "At index %u, expected %u but got %u", i, expected[i], arr[i]);
    }
}

Test(trim_adj_dupes, trim_adj_dupes_test_case_1) {
    int arr[] = {7, 7, 8, 9, 10, 10, 12};
    unsigned int expected[] = {7, 8, 9, 10, 12}; 
    unsigned int reslen = 0;
    int *result = trim_adj_dupes(arr, sizeof(arr) / sizeof(arr[0]), &reslen);
    cr_assert_eq(reslen, sizeof(expected) / sizeof(expected[0]), "Expected length %lu, but got %u", sizeof(expected) / sizeof(expected[0]), reslen);
    for (unsigned int i = 0; i < reslen; i++) {
        cr_assert_eq(result[i], expected[i], "At index %u, expected %u but got %u", i, expected[i], result[i]);
    }
    free(result);
}

Test(trim_adj_dupes, trim_adj_dupes_test_case_2) {
    int arr[] = {8, 8, 9, 12, 1515, 1515, 1515, 1515, 7};
    unsigned int expected[] = {8, 9, 12, 1515, 7}; 
    unsigned int reslen = 0;
    int *result = trim_adj_dupes(arr, sizeof(arr) / sizeof(arr[0]), &reslen);
    cr_assert_eq(reslen, sizeof(expected) / sizeof(expected[0]), "Expected length %lu, but got %u", sizeof(expected) / sizeof(expected[0]), reslen);
    for (unsigned int i = 0; i < reslen; i++) {
        cr_assert_eq(result[i], expected[i], "At index %u, expected %u but got %u", i, expected[i], result[i]);
    }
    free(result);
}

Test(slicer, slicer_test_case_1) {
    int source[] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
    unsigned int indices[] = {3, 5, 7, 9, 5, 3, 0, 4};
    unsigned int reslen = 0;  
    int* result = slicer(source, sizeof(source) / sizeof(source[0]), indices, sizeof(indices) / sizeof(indices[0]), &reslen);
        unsigned int expected[] = {40, 50, 80, 90, 10, 20, 30, 40};
    cr_assert_eq(reslen, sizeof(expected) / sizeof(expected[0]), "Expected  %lu length, got %u", sizeof(expected) / sizeof(expected[0]), reslen);
    for (unsigned int i = 0; i < reslen; i++) {
        cr_assert_eq(result[i], expected[i], "At index %u, expected %u but got %u", i, expected[i], result[i]);
    }
    free(result);
}