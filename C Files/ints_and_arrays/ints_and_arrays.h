/**
 * Checks if a number is ofty, ie., if it contains at 
 * least one copy of each of the digits 1, 4, and 3.
 */
int ofty(unsigned int a);

/**
 * Checks if all elements of the array a with length alen 
 * are ofty
 */
int all_ofty(unsigned int a[], unsigned int alen);

/**
 * Checks if and only if at least one element of the array 
 * is ofty
 */
int exists_ofty(unsigned int a[], unsigned int alen);

/**
 * Returns the index of the first element of the array
 * that's ofty
 */
int first_ofty(unsigned int a[], unsigned int alen);

/**
 * Counts the number of ofty numbers in the array
 */
unsigned int number_ofty(unsigned int a[], unsigned int alen);

/**
 * Adds 1 to any number in an array that's between
 * 141 and 143 inclusive, and changes any other 
 * element not in that range to 0
 */
void progress(unsigned int a[], unsigned int alen);

/**
 * Creates a new array of the same length as the original
 * that provides a running count of the number of ofty 
 * numbers up to each point in the original array
 */
unsigned int* running_ofty(unsigned int a[], unsigned int alen);

/**
 * Shifts the array to the 'right', so that the last element 
 * “wraps around” to become the new first element 
 */
void rotate_right(int a[], unsigned int alen);

/**
 * Trims down array to remove consecutive duplicate elements 
 */
int* trim_adj_dupes(int* a, unsigned int alen, unsigned int* reslen);

/**
 * From a given array, takes all the indicated slices-from 
 * another array containing indices-and concatenate them 
 * together
 */
int* slicer(int* source, unsigned int sourcelen, unsigned int* indices,
            unsigned int indiceslen, unsigned int* reslen);
