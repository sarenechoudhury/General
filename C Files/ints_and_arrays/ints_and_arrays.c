#include <criterion/criterion.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "hw3.h"  

int ofty(unsigned int a){
    char str[12];
    sprintf(str, "%u", a);
    if (strchr(str, '1') != NULL && strchr(str, '4') != NULL && strchr(str, '3') != NULL){
        return 1;
    }
    return 0;

}

int all_ofty(unsigned int a[], unsigned int alen){
    for (int i = 0; i < alen; i++){
        if (ofty(a[i]) == 0){
            return 0;
        }
    }
    return 1;
}

int exists_ofty(unsigned int a[], unsigned int alen){
    for (int i = 0; i < alen; i++){
        if (ofty(a[i]) == 1){
            return 1;
        }
    }
    return 0;
}

int first_ofty(unsigned int a[], unsigned int alen){
    for (int i = 0; i < alen; i++){
        if (ofty(a[i]) == 1){
            return i;
        }
    }
    return -1;
}

unsigned int number_ofty(unsigned int a[], unsigned int alen){
    unsigned int count = 0;
    for (int i = 0; i < alen; i++){
        if (ofty(a[i]) == 1){
            count++;
        }
    }
    return count;
}

void progress(unsigned int a[], unsigned int alen){
    for (int i = 0; i < alen; i++){
        if (a[i] < 140){
            a[i] = 0;
        }
        else if (a[i] > 143){
            a[i] = 0;
        }
        else{
            a[i] = a[i] + 1;
        }
    }
}

unsigned int* running_ofty(unsigned int a[], unsigned int alen){
    unsigned int* x = malloc(alen * sizeof(unsigned int));
    unsigned int count = 0;
    for (int i = 0; i < alen; i++){
        if (ofty(a[i]) == 1){
            count++;
        }
        x[i] = count;
    }
    return x;
    
}

void rotate_right(int a[], unsigned int alen){
    int y = a[alen-1];
    for (int i = alen; i > 0; i--) {
        a[i] = a[i - 1];
    }
    a[0] = y;
}

int* trim_adj_dupes(int* a, unsigned int alen, unsigned int* reslen){
    if (alen == 0){
        *reslen = 0;
        return NULL;
    }
    int *x = (int *)malloc(alen * sizeof(int));
    unsigned int j = 0;
    x[j++] = a[0];
    for (unsigned int i = 1; i < alen; i++) {
        if (a[i] != a[i - 1]) {
            x[j++] = a[i];
        }
    }
    *reslen = j;
    x = (int *)realloc(x, j * sizeof(int));
    return x;
}

int* slicer(int* source, unsigned int sourcelen, unsigned int* indices,
            unsigned int indiceslen, unsigned int* reslen){
                if (indiceslen % 2 != 0) {
                return NULL;  
            }
            int* result = (int*)malloc(sourcelen * sizeof(int));  
            unsigned int result_index = 0; 
            for (unsigned int i = 0; i < indiceslen; i += 2) {
                unsigned int start = indices[i];
                unsigned int end = indices[i + 1];
                if (start >= sourcelen || (end > sourcelen && end != start)) {
                    free(result);
                    return NULL;  
                }
                if (end > sourcelen) {
                    end = sourcelen;
                }
                for (unsigned int j = start; j < end; j++) {
                    result[result_index++] = source[j];
                }
            }
            result = (int*)realloc(result, result_index * sizeof(int));
            if (result == NULL) {
                return NULL;  
            }
            *reslen = result_index;
            return result;
            }